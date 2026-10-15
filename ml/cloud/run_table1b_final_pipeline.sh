#!/usr/bin/env bash
# Durable GPU pipeline: Step1 sanity → Step2 full eval (local+baseline k=1) →
# Step3 noise_k ∈ {0.1,0.3,3} train+select → optional winner eval → table1b → stop VM.
# Upload to GCS after each major step. One infra retry allowed per step.
set -euo pipefail
cd "${REMOTE_DIR:-/home/octav/poliphysics}"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-poliphysics-ballooning-ml}"
BUCKET="${BUCKET:-gs://authorship-verification-poliphysics/tema1-ml}"
ZONE="${ZONE:-us-central1-b}"
INSTANCE="${INSTANCE:-poliphysics-tema1-ml-train}"
LOCAL_RID="gns_local_s42_88dc163f_1791760366"
BASE_RID="gns_baseline_s42_5716ee5b_1791828699"
LOCAL_CKPT="Tema 1/results/ml/runs/${LOCAL_RID}/best.pt"
BASE_CKPT="Tema 1/results/ml/runs/${BASE_RID}/epoch_06.pt"
LOGDIR="Tema 1/results/ml/logs/table1b_final"
mkdir -p "${LOGDIR}" "Tema 1/results/ml/paper_report_pull/report"
MASTER="${LOGDIR}/master.log"
exec > >(tee -a "${MASTER}") 2>&1

echo "=== PIPELINE START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
T_PIPELINE0=$(date +%s)

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY="$(gcloud secrets versions access latest --secret=wandb-api-key 2>/dev/null || true)"
  export WANDB_API_KEY
fi
export WANDB_MODE="${WANDB_MODE:-online}"
export COST_CAP_USD="${COST_CAP_USD:-50}"
export VM_USD_PER_HOUR="${VM_USD_PER_HOUR:-0.853624312}"
export TRAIN_SESSION_T0="${TRAIN_SESSION_T0:-$(date +%s)}"

retry_once() {
  local tag="$1"; shift
  if "$@"; then return 0; fi
  echo "RETRY_ONCE ${tag} after failure @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if "$@"; then return 0; fi
  echo "FATAL ${tag} failed after one retry — stopping for future work"
  gcloud compute instances stop "${INSTANCE}" --zone="${ZONE}" || true
  exit 2
}

upload_path() {
  local src="$1" dst="$2"
  if [[ -e "${src}" ]]; then
    gsutil -m rsync -r "${src}" "${dst}" || gsutil cp -r "${src}" "${dst}" || true
  fi
}

# ---------- Step 1: sanity ----------
echo "=== STEP1 sanity @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
T0=$(date +%s)
retry_once step1_pytest python3 -m pytest ml/tests/ -q --tb=line
# Oracle: prefer after_fix run-ids json if present, else paper_report diagnose
ORACLE_IDS="Tema 1/results/ml/paper_report_pull/diagnostics/gns_baseline_epoch06_val20/summary.json"
if [[ ! -f "${ORACLE_IDS}" ]]; then
  ORACLE_IDS="Tema 1/results/ml/diagnostics/gns_baseline_epoch06_val20_after_fix/summary.json"
fi
# Build run-ids json on the fly from stratified val if missing
if [[ ! -f "${ORACLE_IDS}" ]]; then
  python3 - <<'PY'
import json
from pathlib import Path
from ml.data.dataset import BallooningRunStore
from ml.data.splits import load_splits
from ml.validation import stratified_val_runs
store=BallooningRunStore(Path("/data/processed/5c60fa669a719404"), cache_shards=2, mmap=True)
splits=load_splits(Path("/data/processed/5c60fa669a719404/splits/splits.json"))
ids=stratified_val_runs(store, splits["val"], n_runs=20, seed=42)
out=Path("Tema 1/results/ml/logs/table1b_final/oracle_val20_ids.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"run_ids": ids}, indent=2))
print(out)
PY
  ORACLE_IDS="Tema 1/results/ml/logs/table1b_final/oracle_val20_ids.json"
fi
retry_once step1_oracle python3 -m ml.scripts.oracle_rollout_test \
  --data-dir /data/processed/5c60fa669a719404 \
  --run-ids-json "${ORACLE_IDS}" \
  --t-end 60 --device cpu \
  --out "Tema 1/results/ml/logs/table1b_final/oracle_sanity_t60.json"
python3 - <<'PY'
import json, sys
r=json.load(open("Tema 1/results/ml/logs/table1b_final/oracle_sanity_t60.json"))
c=r["oracle_clean"]
ok = c["n_outcome_match"]==20 and c["n_zero_edge_failures"]==20 and c["max_abs_err_m_worst"]<1e-5
print("ORACLE_OK", ok, c)
sys.exit(0 if ok else 1)
PY
echo "STEP1_WALL_S=$(( $(date +%s) - T0 ))"
upload_path "Tema 1/results/ml/logs/table1b_final" "${BUCKET}/logs/table1b_final/"
echo "STEP1_DONE"

# ---------- Step 2: full test eval ----------
echo "=== STEP2 full eval @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
T0=$(date +%s)
eval_one() {
  local CFG="$1" CKPT="$2" TAG="$3"
  echo "--- EVAL ${TAG} ckpt=${CKPT} ---"
  local t0=$(date +%s)
  python3 -m ml.evaluate --config "${CFG}" --checkpoint "${CKPT}" --device cuda \
    --splits test_interp test_unseenN_interp test_unseenN test_unseenSigma
  echo "EVAL_WALL_S_${TAG}=$(( $(date +%s) - t0 ))"
}
retry_once step2_local eval_one ml/configs/gns_local_gpu.yaml "${LOCAL_CKPT}" gns_local
# sync local eval
RID="${LOCAL_RID}"
if [[ -d "Tema 1/results/ml/${RID}" ]]; then
  upload_path "Tema 1/results/ml/${RID}" "${BUCKET}/runs/${RID}/eval/"
fi
upload_path "Tema 1/results/ml/runs/${RID}" "${BUCKET}/runs/${RID}/"

retry_once step2_baseline eval_one ml/configs/gns_baseline_gpu.yaml "${BASE_CKPT}" gns_baseline
RID="${BASE_RID}"
if [[ -d "Tema 1/results/ml/${RID}" ]]; then
  upload_path "Tema 1/results/ml/${RID}" "${BUCKET}/runs/${RID}/eval/"
fi
upload_path "Tema 1/results/ml/runs/${RID}" "${BUCKET}/runs/${RID}/"
echo "STEP2_WALL_S=$(( $(date +%s) - T0 ))"
echo "STEP2_DONE"

# ---------- Step 3: noise_k sweep ----------
echo "=== STEP3 noise_k sweep @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
T0=$(date +%s)
# Patch configs for each k (same as baseline gpu otherwise)
for K in 0.1 0.3 3; do
  CFG="Tema 1/results/ml/logs/table1b_final/gns_baseline_k${K}.yaml"
  python3 - <<PY
import yaml
from pathlib import Path
cfg=yaml.safe_load(Path("ml/configs/gns_baseline_gpu.yaml").read_text())
cfg["data"]["noise_k"]=float("${K}")
# Ensure early stopping patience 2
cfg.setdefault("train",{})
cfg["train"].setdefault("early_stopping",{})
cfg["train"]["early_stopping"]={"metric":"val_rollout_mse","patience_epochs":2}
cfg["train"]["cost_cap_usd"]=float("${COST_CAP_USD}")
cfg["train"]["vm_usd_per_hour"]=float("${VM_USD_PER_HOUR}")
Path("${CFG}").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("wrote", "${CFG}")
PY
  echo "--- TRAIN noise_k=${K} ---"
  export TRAIN_SESSION_T0=$(date +%s)
  # epoch watcher in background
  RUN_GLOB="Tema 1/results/ml/runs/gns_baseline_s42_*"
  LOG="Tema 1/results/ml/logs/table1b_final/train_k${K}.log"
  # Start train; capture run dir from output
  set +e
  python3 -m ml.train --config "${CFG}" --device cuda 2>&1 | tee "${LOG}"
  TRC=${PIPESTATUS[0]}
  set -e
  # Find newest baseline run dir
  RUN_DIR=$(ls -dt Tema\ 1/results/ml/runs/gns_baseline_s42_* 2>/dev/null | head -1)
  echo "TRAIN_K${K}_EXIT=${TRC} RUN_DIR=${RUN_DIR}"
  # Accept early_stop (0) or cost_cap (43) if we got checkpoints
  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "FATAL no run dir for k=${K}"
    exit 2
  fi
  upload_path "${RUN_DIR}" "${BUCKET}/runs/$(basename "${RUN_DIR}")/"
  # Select on epoch_*.pt up to stop (all present)
  CKPTS=()
  shopt -s nullglob
  for f in "${RUN_DIR}"/epoch_*.pt; do CKPTS+=("$f"); done
  [[ -f "${RUN_DIR}/best.pt" ]] && CKPTS+=("${RUN_DIR}/best.pt")
  shopt -u nullglob
  SEL_OUT="Tema 1/results/ml/logs/table1b_final/ckpt_select_k${K}.json"
  python3 -m ml.scripts.select_checkpoint_horizon \
    --config "${CFG}" \
    --checkpoints "${CKPTS[@]}" \
    --device cuda --n-runs 32 --t-end 60 \
    --out "${SEL_OUT}"
  upload_path "${SEL_OUT}" "${BUCKET}/logs/table1b_final/"
done

# k=1 select restricted (all epochs 00-10; early stop not reached)
echo "--- SELECT k=1 post-hoc restriction ---"
K1_CKPTS=()
shopt -s nullglob
for f in "Tema 1/results/ml/runs/${BASE_RID}"/epoch_0{0,1,2,3,4,5,6,7,8,9}.pt \
         "Tema 1/results/ml/runs/${BASE_RID}"/epoch_10.pt; do
  [[ -f "$f" ]] && K1_CKPTS+=("$f")
done
[[ -f "Tema 1/results/ml/runs/${BASE_RID}/best.pt" ]] && K1_CKPTS+=("Tema 1/results/ml/runs/${BASE_RID}/best.pt")
shopt -u nullglob
python3 -m ml.scripts.select_checkpoint_horizon \
  --config ml/configs/gns_baseline_gpu.yaml \
  --checkpoints "${K1_CKPTS[@]}" \
  --device cuda --n-runs 32 --t-end 60 \
  --out "Tema 1/results/ml/logs/table1b_final/ckpt_select_k1.json"
upload_path "Tema 1/results/ml/logs/table1b_final" "${BUCKET}/logs/table1b_final/"

# Build selection table + pick winner
python3 - <<'PY'
import json
from pathlib import Path
rows=[]
# k=1
d=json.load(open("Tema 1/results/ml/logs/table1b_final/ckpt_select_k1.json"))
ph=json.load(open("Tema 1/results/ml/logs/k1_posthoc_early_stop.json"))
sel=d["selected"]; row=d["checkpoints"][sel]
rows.append({
  "k":1.0,
  "selected":sel,
  "path":row.get("path"),
  "ttf_median":row["ttf"]["median"],
  "ttf_q25":row["ttf"]["q25"],
  "ttf_q75":row["ttf"]["q75"],
  "never_fail_frac":row["ttf"]["n_never_fail"]/max(row["ttf"]["n_runs"],1),
  "capped_5s_mse":row.get("select_5s_rel_mse"),
  "phys_frac_5s": (row.get("physical_fraction_at_times") or {}).get("5s") or (row.get("physical_fraction_at_times") or {}).get("5.0"),
  "epochs_trained": ph["epochs_trained_observed"],
  "stop_epoch": ph["post_hoc_early_stop_epoch"],
  "best_val_metric": ph["best_val_rollout_mse"],
  "best_epoch": ph["best_epoch"],
})
for k in (0.1, 0.3, 3.0):
  p=Path(f"Tema 1/results/ml/logs/table1b_final/ckpt_select_k{k}.json")
  if not p.exists():
    # try without trailing .0
    p=Path(f"Tema 1/results/ml/logs/table1b_final/ckpt_select_k{k:g}.json")
  d=json.load(open(p))
  sel=d["selected"]; row=d["checkpoints"][sel]
  # read run_meta from path parent
  from pathlib import Path as P
  meta={}
  try:
    meta=json.load(open(P(row["path"]).parent/"run_meta.json"))
  except Exception:
    pass
  rows.append({
    "k":k,
    "selected":sel,
    "path":row.get("path"),
    "ttf_median":row["ttf"]["median"],
    "ttf_q25":row["ttf"]["q25"],
    "ttf_q75":row["ttf"]["q75"],
    "never_fail_frac":row["ttf"]["n_never_fail"]/max(row["ttf"]["n_runs"],1),
    "capped_5s_mse":row.get("select_5s_rel_mse"),
    "phys_frac_5s": (row.get("physical_fraction_at_times") or {}).get("5s") or (row.get("physical_fraction_at_times") or {}).get("5.0"),
    "epochs_trained": meta.get("epochs_completed"),
    "stop_epoch": meta.get("epochs_completed"),
    "best_val_metric": meta.get("best_rollout_mse"),
    "best_epoch": meta.get("best_epoch"),
    "stop_reason": meta.get("stop_reason"),
  })
# Winner: max median TTF, then min capped 5s
best=None
for r in rows:
  if best is None:
    best=r; continue
  if r["ttf_median"]>best["ttf_median"]+1e-12 or (
      abs(r["ttf_median"]-best["ttf_median"])<=1e-12 and r["capped_5s_mse"]<best["capped_5s_mse"]):
    best=r
out={"rows":rows,"winner":best}
Path("Tema 1/results/ml/logs/table1b_final/noise_k_selection_table.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
# flag if winner k != 1
Path("Tema 1/results/ml/logs/table1b_final/winner_k.txt").write_text(str(best["k"]))
Path("Tema 1/results/ml/logs/table1b_final/winner_ckpt.txt").write_text(best["path"] or "")
PY

WIN_K=$(cat "Tema 1/results/ml/logs/table1b_final/winner_k.txt")
WIN_CKPT=$(cat "Tema 1/results/ml/logs/table1b_final/winner_ckpt.txt")
echo "WINNER_K=${WIN_K} CKPT=${WIN_CKPT}"
if python3 -c "import sys; sys.exit(0 if abs(float('${WIN_K}')-1.0)>1e-9 else 1)"; then
  echo "--- EVAL winning k=${WIN_K} ---"
  # find matching config
  CFG="Tema 1/results/ml/logs/table1b_final/gns_baseline_k${WIN_K}.yaml"
  [[ -f "${CFG}" ]] || CFG="Tema 1/results/ml/logs/table1b_final/gns_baseline_k${WIN_K%.*}.yaml"
  retry_once step3_winner_eval eval_one "${CFG}" "${WIN_CKPT}" "gns_baseline_k${WIN_K}"
fi
echo "STEP3_WALL_S=$(( $(date +%s) - T0 ))"
echo "STEP3_DONE"
upload_path "Tema 1/results/ml/logs/table1b_final" "${BUCKET}/logs/table1b_final/"

# ---------- Step 4: assemble table1b (CV/FA from existing v2 + GNS) ----------
echo "=== STEP4 assemble table @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 -m ml.scripts.assemble_table1b_v2_final \
  --out-dir "Tema 1/results/ml/paper_report_pull/report" \
  --pipeline-wall-s "$(( $(date +%s) - T_PIPELINE0 ))" \
  || python3 - <<'PY'
# Fallback assembler if script missing
print("assemble script missing — write stub summary")
from pathlib import Path
import json, time
out=Path("Tema 1/results/ml/paper_report_pull/report")
out.mkdir(parents=True, exist_ok=True)
(out/"summary.md").write_text(
  "# Table 1b v2 final — partial\n\nAssembler not present; see logs/table1b_final/.\n"
)
PY
upload_path "Tema 1/results/ml/paper_report_pull/report" "${BUCKET}/paper_report_pull/report/"
upload_path "Tema 1/results/ml/logs/table1b_final" "${BUCKET}/logs/table1b_final/"

echo "=== PIPELINE DONE wall_s=$(( $(date +%s) - T_PIPELINE0 )) @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
gcloud compute instances stop "${INSTANCE}" --zone="${ZONE}" || true
echo VM_STOPPED
