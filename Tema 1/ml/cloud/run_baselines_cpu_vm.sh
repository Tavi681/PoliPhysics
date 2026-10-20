#!/usr/bin/env bash
# CV + follow-air paper eval on a CPU VM (massless-tracer FollowAir, schema v3).
# Writes results/ml/eval_v3/ and gs://…/tema1-ml/eval_v3/ only.
# Does not create or delete the VM. Caller mounts /data/processed/<data version>.
set -euo pipefail
cd "${REMOTE_DIR:-/home/octav/poliphysics}/Tema 1"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
BUCKET="${BUCKET:-gs://authorship-verification-poliphysics/tema1-ml}"
DATA_VER="${DATA_VER:-5c60fa669a719404}"
PROCESSED="/data/processed/${DATA_VER}"
LOG_DIR="results/ml/logs/baselines_cpu_v3"
CV_RUN="_baseline_const_cpu_v3"
FA_RUN="_baseline_follow_air_cpu_v3"
CV_CFG="${LOG_DIR}/baseline_const_cpu_v3.yaml"
FA_CFG="${LOG_DIR}/baseline_follow_air_cpu_v3.yaml"
SANITY_IDS="results/ml/diagnostics/gns_baseline_epoch06_val20_after_fix/summary.json"
# Fall back to the paper_report_pull copy if the diagnostics tree was not synced.
[[ -f "${SANITY_IDS}" ]] || SANITY_IDS="results/ml/paper_report_pull/diagnostics/gns_baseline_epoch06_val20/summary.json"
CV_WALL_CAP_S=$((2 * 3 * 3600 + 2 * 720))   # 2× ~3.2 h
FA_WALL_CAP_S=$((2 * 9 * 3600))              # 2× ~9 h

mkdir -p "${LOG_DIR}" "results/ml/runs" "results/ml/eval_v3/constant_velocity/per_run" \
  "results/ml/eval_v3/follow_air/per_run" "results/ml/diagnostics"

echo "=== baselines CPU @ $(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$(cat ml/COMMIT 2>/dev/null || echo unknown) ==="

if [[ ! -f "${PROCESSED}/splits/splits.json" ]]; then
  echo "SYNC processed ${DATA_VER} from GCS..."
  mkdir -p "${PROCESSED}"
  gcloud storage rsync --recursive "${BUCKET}/processed/${DATA_VER}" "${PROCESSED}" \
    || gsutil -m rsync -r "${BUCKET}/processed/${DATA_VER}/" "${PROCESSED}/"
fi
[[ -f "${PROCESSED}/splits/splits.json" ]] || { echo "FATAL missing splits"; exit 1; }
echo "DATA_OK ${PROCESSED}"

echo "=== STEP2a pytest ==="
T0=$(date +%s)
python3 -m pytest ml/tests -q --tb=line 2>&1 | tee "${LOG_DIR}/pytest_ml.log"
python3 -m pytest ml/tests/test_oracle_rollout.py -q --tb=line -rs 2>&1 | tee "${LOG_DIR}/pytest_oracle.log"
echo "STEP2a_WALL_S=$(( $(date +%s) - T0 ))"

# Configs: same metrics as GNS paper eval; per_run dirs are eval_v3 (never the pre-fix cache).
python3 - <<'PY'
from pathlib import Path
import yaml
log = Path("results/ml/logs/baselines_cpu_v3")
for src, dst, model, per in (
    ("ml/configs/baseline_const_cpu.yaml", log / "baseline_const_cpu_v3.yaml",
     "constant_velocity", "results/ml/eval_v3/constant_velocity/per_run"),
    ("ml/configs/baseline_follow_air_cpu.yaml", log / "baseline_follow_air_cpu_v3.yaml",
     "follow_air", "results/ml/eval_v3/follow_air/per_run"),
):
    cfg = yaml.safe_load(Path(src).read_text())
    cfg["model"]["name"] = model
    cfg["eval"] = dict(cfg.get("eval") or {})
    cfg["eval"]["per_run_dir"] = per
    cfg["eval"]["stream_full_rollouts"] = True
    cfg["eval"]["t_end"] = 60.0
    cfg["eval"]["seed"] = 42
    cfg["eval"]["one_step_n_frames"] = 100000
    cfg["eval"]["horizon_errors_s"] = [0.2, 1.0, 5.0, 10.0]
    cfg["eval"]["error_vs_time_grid_s"] = None
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("wrote", dst, "per_run_dir", per)
PY

make_baseline_ckpt() {
  local NAME="$1" CFG="$2" OUTDIR="$3"
  python3 - <<PY
from pathlib import Path
import torch
from ml.utils.config import load_yaml
from ml.train import _resolve
from ml.data.dataset import BallooningRunStore
from ml.models import build_model
cfg = load_yaml(_resolve("${CFG}"))
dd = cfg["data"]["data_dir"]
store = BallooningRunStore(dd if Path(dd).is_absolute() else _resolve(dd))
m = build_model("${NAME}", a_mean=store.norm["a"]["mean"], a_std=store.norm["a"]["std"])
out = Path("${OUTDIR}")
out.mkdir(parents=True, exist_ok=True)
torch.save({"model": m.state_dict(), "meta": {"run_id": out.name, "train_time_s": 0.0}, "cfg": cfg}, out / "best.pt")
print(out / "best.pt")
PY
}

CV_CKPT=$(make_baseline_ckpt constant_velocity "${CV_CFG}" "results/ml/runs/${CV_RUN}")
FA_CKPT=$(make_baseline_ckpt follow_air "${FA_CFG}" "results/ml/runs/${FA_RUN}")
echo "CV_CKPT=${CV_CKPT} FA_CKPT=${FA_CKPT}"

echo "=== STEP2b sanity 20 val runs ==="
T0=$(date +%s)
python3 - <<PY
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from ml.utils.config import load_yaml
from ml.train import _resolve, _device
from ml.data.dataset import BallooningRunStore
from ml.models import build_model
from ml.evaluate import full_rollout_surrogate

ids_path = Path("${SANITY_IDS}")
if not ids_path.is_file():
    raise SystemExit(f"missing sanity ids: {ids_path}")
run_ids = json.loads(ids_path.read_text())["run_ids"]
assert len(run_ids) == 20, len(run_ids)

def load_model(cfg_path, ckpt_path, name):
    cfg = load_yaml(_resolve(cfg_path))
    dd = cfg["data"]["data_dir"]
    store = BallooningRunStore(dd if Path(dd).is_absolute() else _resolve(dd))
    m = build_model(name, a_mean=store.norm["a"]["mean"], a_std=store.norm["a"]["std"])
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(blob["model"], strict=False)
    m.to(_device("cpu")).eval()
    return m, store, cfg

def run_sanity(tag, cfg_path, ckpt_path, name):
    model, store, cfg = load_model(cfg_path, ckpt_path, name)
    dcfg = cfg["data"]
    rows = []
    for rid in run_ids:
        rec = store.get(rid)
        pred = full_rollout_surrogate(
            model, rec, store.norm, _device("cpu"),
            C=int(dcfg.get("C", 5)), t_end=60.0,
            node_features=dcfg.get("node_features"),
            globals_fields=dcfg.get("globals_fields"),
            edge_sets=dcfg.get("edge_sets"),
            radius=dcfg.get("radius", 0.1),
            normalize_features=dcfg.get("normalize_features", True),
            return_trajectory=False,
            horizon_errors_s=list(cfg.get("eval", {}).get("horizon_errors_s", [0.2, 1.0, 5.0, 10.0])),
        )
        rows.append({
            "run_id": rid,
            "outcome_sim": pred.get("outcome_sim"),
            "status": pred.get("status"),
            "ttf_s": pred.get("ttf_s"),
            "fail_node_i": pred.get("fail_node_i"),
            "fail_node_j": pred.get("fail_node_j"),
            "exit_time": pred.get("exit_time"),
        })
        print(f"{tag} {rid} sim={pred.get('outcome_sim')} surr={pred.get('status')} ttf={pred.get('ttf_s')}", flush=True)
    ttfs = np.asarray([float(r["ttf_s"]) for r in rows], dtype=np.float64)
    finite = ttfs[np.isfinite(ttfs)]
    out = {
        "tag": tag,
        "n": len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "outcome_sim_counts": dict(Counter(r["outcome_sim"] for r in rows)),
        "ttf_median": float(np.median(finite)) if finite.size else float("inf"),
        "n_timeout": sum(1 for r in rows if r["status"] == "timeout"),
        "first_fail_edges": [
            {"run_id": r["run_id"], "i": r["fail_node_i"], "j": r["fail_node_j"], "ttf_s": r["ttf_s"]}
            for r in rows if r["fail_node_i"] is not None
        ],
        "rows": rows,
    }
    path = Path("results/ml/logs/baselines_cpu_v3") / f"sanity_20_{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if k != "rows"}, indent=2))
    return out

cv = run_sanity("CV", "${CV_CFG}", "${CV_CKPT}", "constant_velocity")
fa = run_sanity("FA", "${FA_CFG}", "${FA_CKPT}", "follow_air")
# Simulator has 0/20 timeouts on this set; FA must not time out on most runs.
if fa["n_timeout"] > 5:
    raise SystemExit(f"STOP: FA timeouts={fa['n_timeout']}/20 (expected small; sim has 0)")
print("SANITY_OK")
PY
echo "STEP2b_WALL_S=$(( $(date +%s) - T0 ))"

eval_one() {
  local CFG="$1" CKPT="$2" TAG="$3" CAP_S="$4" GCS_MODEL="$5"
  echo "=== FULL EVAL ${TAG} @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  local T0=$(date +%s)
  # Watchdog: if wall exceeds 2× prior estimate, kill evaluate.
  (
    sleep "${CAP_S}"
    echo "WATCHDOG_KILL ${TAG} after ${CAP_S}s" | tee -a "${LOG_DIR}/watchdog.log"
    pkill -f "ml.evaluate --config ${CFG}" || true
  ) &
  local WD=$!
  set +e
  /usr/bin/time -v python3 -m ml.evaluate --config "${CFG}" --checkpoint "${CKPT}" --device cpu \
    --splits test_interp test_unseenN_interp test_unseenN test_unseenSigma \
    2>&1 | tee "${LOG_DIR}/eval_${TAG}.log"
  local RC=${PIPESTATUS[0]}
  set -e
  kill "${WD}" 2>/dev/null || true
  wait "${WD}" 2>/dev/null || true
  local T1=$(date +%s)
  local WALL=$((T1 - T0))
  echo "EVAL_WALL_S_${TAG}=${WALL}" | tee -a "${LOG_DIR}/wall_times.txt"
  if [[ ${WALL} -ge ${CAP_S} ]]; then
    echo "FATAL ${TAG} hit wall cap ${CAP_S}s" >&2
    exit 2
  fi
  [[ ${RC} -eq 0 ]] || { echo "FATAL evaluate ${TAG} rc=${RC}" >&2; exit ${RC}; }
  local RID
  RID=$(python3 -c "import torch; b=torch.load('${CKPT}', map_location='cpu', weights_only=False); print((b.get('meta') or {}).get('run_id') or '')")
  local EVAL_DIR="results/ml/${RID}"
  gsutil -m rsync -r "${EVAL_DIR}" "${BUCKET}/eval_v3/${GCS_MODEL}/" || true
  gsutil -m rsync -r "results/ml/eval_v3/${GCS_MODEL}/per_run" "${BUCKET}/eval_v3/${GCS_MODEL}/per_run/" || true
  echo "GCS_SYNCED eval_v3/${GCS_MODEL}"
}

eval_one "${CV_CFG}" "${CV_CKPT}" constant_velocity "${CV_WALL_CAP_S}" constant_velocity
eval_one "${FA_CFG}" "${FA_CKPT}" follow_air "${FA_WALL_CAP_S}" follow_air

gsutil -m rsync -r "${LOG_DIR}" "${BUCKET}/eval_v3/logs/" || true
echo "=== DONE_BASELINES $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
