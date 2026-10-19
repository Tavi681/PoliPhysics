#!/usr/bin/env bash
# Step3: train gns_baseline for noise_k in {0.1, 0.3, 3}, select (max TTF then min 5s),
# re-select k=1 under post-hoc epoch window, eval winner if !=1, assemble table, stop VM.
set -euo pipefail
cd "${REMOTE_DIR:-/home/octav/poliphysics}/Tema 1"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-poliphysics-ballooning-ml}"
BUCKET="${BUCKET:-gs://authorship-verification-poliphysics/tema1-ml}"
ZONE="${ZONE:-us-central1-b}"
INSTANCE="${INSTANCE:-poliphysics-tema1-ml-train}"
LOGDIR="results/ml/logs/table1b_final"
BASE_RID="gns_baseline_s42_5716ee5b_1791828699"
mkdir -p "${LOGDIR}"
exec > >(tee -a "${LOGDIR}/step3.log") 2>&1

echo "=== STEP3 START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
T0_ALL=$(date +%s)

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY="$(gcloud secrets versions access latest --secret=wandb-api-key 2>/dev/null || true)"
  export WANDB_API_KEY
fi
export WANDB_MODE="${WANDB_MODE:-online}"
# High enough that patience-2 early stop can fire (~12 epochs × ~3 h × $0.85 ≈ $30; use 50).
export COST_CAP_USD="${COST_CAP_USD:-50}"
export VM_USD_PER_HOUR="${VM_USD_PER_HOUR:-0.853624312}"

retry_once() {
  local tag="$1"; shift
  if "$@"; then return 0; fi
  echo "RETRY_ONCE ${tag}"
  if "$@"; then return 0; fi
  echo "FATAL ${tag}"; gcloud compute instances stop "${INSTANCE}" --zone="${ZONE}" || true; exit 2
}

upload() {
  local src="$1" dst="$2"
  [[ -e "${src}" ]] || return 0
  gsutil -m rsync -r "${src}" "${dst}" || true
}

train_one_k() {
  local K="$1"
  local CFG="${LOGDIR}/gns_baseline_k${K}.yaml"
  python3 - <<PY
import yaml
from pathlib import Path
cfg=yaml.safe_load(Path("ml/configs/gns_baseline_gpu.yaml").read_text())
cfg["data"]["noise_k"]=float("${K}")
cfg.setdefault("train", {})
cfg["train"]["early_stopping"]={"metric":"val_rollout_mse","patience_epochs":2}
cfg["train"]["cost_cap_usd"]=float("${COST_CAP_USD}")
cfg["train"]["vm_usd_per_hour"]=float("${VM_USD_PER_HOUR}")
Path("${CFG}").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("wrote", "${CFG}")
PY
  local TRAIN_LOG="${LOGDIR}/train_k${K}.log"
  export TRAIN_SESSION_T0=$(date +%s)
  # Snapshot existing run dirs
  python3 - <<'PY'
from pathlib import Path
Path("results/ml/logs/table1b_final/_runs_before.txt").write_text(
  "\n".join(sorted(p.name for p in Path("results/ml/runs").glob("gns_baseline_s42_*")))
)
PY
  echo "--- TRAIN k=${K} @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
  set +e
  python3 -m ml.train --config "${CFG}" --device cuda > "${TRAIN_LOG}" 2>&1 &
  local TRAIN_PID=$!
  set -e
  # Wait for new run dir
  local RUN_DIR=""
  for _ in $(seq 1 120); do
    sleep 5
    RUN_DIR=$(python3 - <<'PY'
from pathlib import Path
before=set(Path("results/ml/logs/table1b_final/_runs_before.txt").read_text().split())
now={p.name for p in Path("results/ml/runs").glob("gns_baseline_s42_*")}
new=sorted(now-before)
print(new[-1] if new else "")
PY
)
    if [[ -n "${RUN_DIR}" && -d "results/ml/runs/${RUN_DIR}" ]]; then
      break
    fi
    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
      echo "train exited before run dir"; break
    fi
  done
  echo "RUN_DIR=${RUN_DIR}"
  if [[ -z "${RUN_DIR}" ]]; then
    echo "FATAL no run dir for k=${K}"
    wait "${TRAIN_PID}" || true
    return 1
  fi
  # Epoch watcher
  nohup bash ml/cloud/watch_epoch_checkpoints.sh \
    --run-dir "results/ml/runs/${RUN_DIR}" \
    --log "${TRAIN_LOG}" \
    > "${LOGDIR}/epoch_watch_k${K}.log" 2>&1 &
  echo "EPOCH_WATCH_PID=$!"
  set +e
  wait "${TRAIN_PID}"
  local TRC=$?
  set -e
  echo "TRAIN_K${K}_EXIT=${TRC}"
  # Allow 0 (early/max) or 43 (cost_cap)
  if [[ ! -f "results/ml/runs/${RUN_DIR}/best.pt" && ! -f "results/ml/runs/${RUN_DIR}/last.pt" ]]; then
    echo "FATAL no checkpoints for k=${K}"
    return 1
  fi
  # Persist mapping
  echo "${RUN_DIR}" > "${LOGDIR}/run_dir_k${K}.txt"
  upload "results/ml/runs/${RUN_DIR}" "${BUCKET}/runs/${RUN_DIR}/"
  # Select
  local CKPTS=()
  shopt -s nullglob
  for f in "results/ml/runs/${RUN_DIR}"/epoch_*.pt; do CKPTS+=("$f"); done
  [[ -f "results/ml/runs/${RUN_DIR}/best.pt" ]] && CKPTS+=("results/ml/runs/${RUN_DIR}/best.pt")
  [[ -f "results/ml/runs/${RUN_DIR}/last.pt" ]] && CKPTS+=("results/ml/runs/${RUN_DIR}/last.pt")
  shopt -u nullglob
  if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "FATAL empty ckpt list k=${K}"; return 1
  fi
  python3 -m ml.scripts.select_checkpoint_horizon \
    --config "${CFG}" \
    --checkpoints "${CKPTS[@]}" \
    --device cuda --n-runs 32 --t-end 60 \
    --out "${LOGDIR}/ckpt_select_k${K}.json"
  upload "${LOGDIR}" "${BUCKET}/logs/table1b_final/"
}

for K in 0.1 0.3 3; do
  if [[ -f "${LOGDIR}/ckpt_select_k${K}.json" && -f "${LOGDIR}/run_dir_k${K}.txt" ]]; then
    echo "SKIP train k=${K} (ckpt_select_k${K}.json exists)"
    continue
  fi
  retry_once "train_select_k${K}" train_one_k "${K}"
done

if [[ -f "${LOGDIR}/ckpt_select_k1.json" ]]; then
  echo "SKIP select k=1 (ckpt_select_k1.json exists)"
else
echo "--- SELECT k=1 (epochs 00-10; post-hoc early stop not reached) ---"
K1_CKPTS=()
shopt -s nullglob
for f in "results/ml/runs/${BASE_RID}"/epoch_*.pt; do K1_CKPTS+=("$f"); done
[[ -f "results/ml/runs/${BASE_RID}/best.pt" ]] && K1_CKPTS+=("results/ml/runs/${BASE_RID}/best.pt")
shopt -u nullglob
python3 -m ml.scripts.select_checkpoint_horizon \
  --config ml/configs/gns_baseline_gpu.yaml \
  --checkpoints "${K1_CKPTS[@]}" \
  --device cuda --n-runs 32 --t-end 60 \
  --out "${LOGDIR}/ckpt_select_k1.json"
fi

python3 - <<'PY'
import json
from pathlib import Path
LOG=Path("results/ml/logs/table1b_final")
rows=[]
ph=json.loads((LOG/"../k1_posthoc_early_stop.json").read_text()) if (LOG/"../k1_posthoc_early_stop.json").exists() else json.loads(Path("results/ml/logs/k1_posthoc_early_stop.json").read_text())
d=json.load(open(LOG/"ckpt_select_k1.json"))
sel=d["selected"]; row=d["checkpoints"][sel]
phys=row.get("physical_fraction_at_times") or {}
rows.append({
  "k":1.0,"selected":sel,"path":row.get("path"),
  "ttf_median":row["ttf"]["median"],"ttf_q25":row["ttf"]["q25"],"ttf_q75":row["ttf"]["q75"],
  "never_fail_frac":row["ttf"]["n_never_fail"]/max(row["ttf"]["n_runs"],1),
  "capped_5s_mse":row.get("select_5s_rel_mse"),
  "phys_frac_5s": phys.get("5s", phys.get("5.0")),
  "epochs_trained": ph.get("epochs_trained_observed"),
  "stop_epoch": ph.get("post_hoc_early_stop_epoch"),
  "best_val_metric": ph.get("best_val_rollout_mse"),
  "best_epoch": ph.get("best_epoch"),
})
for k in (0.1, 0.3, 3.0):
  # bash writes k=3 as ckpt_select_k3.json; f"{3.0}" is ckpt_select_k3.0.json
  p=LOG/f"ckpt_select_k{k:g}.json"
  if not p.exists():
    p=LOG/f"ckpt_select_k{k}.json"
  d=json.load(open(p))
  sel=d["selected"]; row=d["checkpoints"][sel]
  run_dir=Path(row["path"]).parent
  meta={}
  mp=run_dir/"run_meta.json"
  if mp.exists():
    meta=json.loads(mp.read_text())
  phys=row.get("physical_fraction_at_times") or {}
  rows.append({
    "k":k,"selected":sel,"path":row.get("path"),
    "ttf_median":row["ttf"]["median"],"ttf_q25":row["ttf"]["q25"],"ttf_q75":row["ttf"]["q75"],
    "never_fail_frac":row["ttf"]["n_never_fail"]/max(row["ttf"]["n_runs"],1),
    "capped_5s_mse":row.get("select_5s_rel_mse"),
    "phys_frac_5s": phys.get("5s", phys.get("5.0")),
    "epochs_trained": meta.get("epochs_completed"),
    "stop_epoch": meta.get("epochs_completed"),
    "best_val_metric": meta.get("best_rollout_mse"),
    "best_epoch": meta.get("best_epoch"),
    "stop_reason": meta.get("stop_reason"),
  })
best=None
for r in rows:
  if best is None:
    best=r; continue
  if (r["ttf_median"]>best["ttf_median"]+1e-12) or (
      abs(r["ttf_median"]-best["ttf_median"])<=1e-12 and r["capped_5s_mse"]<best["capped_5s_mse"]):
    best=r
out={"rows":rows,"winner":best}
(LOG/"noise_k_selection_table.json").write_text(json.dumps(out, indent=2))
(LOG/"winner_k.txt").write_text(f"{best['k']:g}")
(LOG/"winner_ckpt.txt").write_text(best["path"] or "")
print(json.dumps(out, indent=2))
PY

WIN_K=$(cat "${LOGDIR}/winner_k.txt")
WIN_CKPT=$(cat "${LOGDIR}/winner_ckpt.txt")
echo "WINNER_K=${WIN_K} CKPT=${WIN_CKPT}"
if python3 -c "import sys; sys.exit(0 if abs(float('${WIN_K}')-1.0)>1e-9 else 1)"; then
  CFG="${LOGDIR}/gns_baseline_k${WIN_K}.yaml"
  retry_once winner_eval python3 -m ml.evaluate \
    --config "${CFG}" --checkpoint "${WIN_CKPT}" --device cuda \
    --splits test_interp test_unseenN_interp test_unseenN test_unseenSigma
  RID=$(basename "$(dirname "${WIN_CKPT}")")
  upload "results/ml/${RID}" "${BUCKET}/runs/${RID}/eval/"
  upload "results/ml/runs/${RID}" "${BUCKET}/runs/${RID}/"
fi

echo "STEP3_WALL_S=$(( $(date +%s)-T0_ALL ))"
upload "${LOGDIR}" "${BUCKET}/logs/table1b_final/"

if [[ "${SKIP_STEP4:-0}" == "1" ]]; then
  echo "SKIP_STEP4 (step3 remainder only)"
  echo "=== STEP3 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  exit 0
fi

echo "=== STEP4 assemble ==="
python3 -m ml.scripts.assemble_table1b_v2_final \
  --out-dir "results/ml/paper_report_pull/report" \
  --pipeline-wall-s "$(( $(date +%s)-T0_ALL ))"
upload "results/ml/paper_report_pull/report" "${BUCKET}/paper_report_pull/report/"

echo "=== STEP3+4 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
gcloud compute instances stop "${INSTANCE}" --zone="${ZONE}" || true
echo VM_STOPPED
