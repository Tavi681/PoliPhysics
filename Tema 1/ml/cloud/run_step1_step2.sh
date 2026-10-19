#!/usr/bin/env bash
# Step1 sanity + Step2 full test eval for gns_local best + gns_baseline epoch_06.
# Uploads to GCS after each step. Does NOT stop the VM (Step3 follows).
set -euo pipefail
cd "${REMOTE_DIR:-/home/octav/poliphysics}/Tema 1"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
BUCKET="${BUCKET:-gs://authorship-verification-poliphysics/tema1-ml}"
LOCAL_RID="gns_local_s42_88dc163f_1791760366"
BASE_RID="gns_baseline_s42_5716ee5b_1791828699"
LOCAL_CKPT="results/ml/runs/${LOCAL_RID}/best.pt"
BASE_CKPT="results/ml/runs/${BASE_RID}/epoch_06.pt"
LOGDIR="results/ml/logs/table1b_final"
mkdir -p "${LOGDIR}"
exec > >(tee -a "${LOGDIR}/step1_step2.log") 2>&1

echo "=== STEP1+2 START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
T0_ALL=$(date +%s)

retry_once() {
  local tag="$1"; shift
  if "$@"; then return 0; fi
  echo "RETRY_ONCE ${tag}"
  if "$@"; then return 0; fi
  echo "FATAL ${tag}"; exit 2
}

upload() {
  local src="$1" dst="$2"
  [[ -e "${src}" ]] || return 0
  gsutil -m rsync -r "${src}" "${dst}" || true
}

echo "=== STEP1 pytest ==="
T0=$(date +%s)
retry_once pytest python3 -m pytest ml/tests/ -q --tb=line

# Oracle run ids
ORACLE_IDS="${LOGDIR}/oracle_val20_ids.json"
python3 - <<PY
import json
from pathlib import Path
from ml.data.dataset import BallooningRunStore
from ml.data.splits import load_splits
from ml.validation import stratified_val_runs
p=Path("results/ml/paper_report_pull/diagnostics/gns_baseline_epoch06_val20/summary.json")
if p.exists():
    ids=json.loads(p.read_text())["run_ids"]
else:
    store=BallooningRunStore(Path("/data/processed/5c60fa669a719404"), cache_shards=2, mmap=True)
    splits=load_splits(Path("/data/processed/5c60fa669a719404/splits/splits.json"))
    ids=stratified_val_runs(store, splits["val"], n_runs=20, seed=42)
Path("${ORACLE_IDS}").write_text(json.dumps({"run_ids": ids}, indent=2))
print("n_ids", len(ids))
PY

echo "=== STEP1 oracle ==="
retry_once oracle python3 -m ml.scripts.oracle_rollout_test \
  --data-dir /data/processed/5c60fa669a719404 \
  --run-ids-json "${ORACLE_IDS}" \
  --t-end 60 --device cpu \
  --out "${LOGDIR}/oracle_sanity_t60.json"
python3 - <<'PY'
import json, sys
r=json.load(open("results/ml/logs/table1b_final/oracle_sanity_t60.json"))
c=r["oracle_clean"]
ok=c["n_outcome_match"]==20 and c["n_zero_edge_failures"]==20 and c["max_abs_err_m_worst"]<1e-5
print("ORACLE", c["n_outcome_match"], c["n_zero_edge_failures"], c["max_abs_err_m_worst"], "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
echo "STEP1_WALL_S=$(( $(date +%s)-T0 ))"
upload "${LOGDIR}" "${BUCKET}/logs/table1b_final/"
echo "STEP1_DONE"

echo "=== STEP2 gns_local ==="
T0=$(date +%s)
retry_once eval_local python3 -m ml.evaluate \
  --config ml/configs/gns_local_gpu.yaml \
  --checkpoint "${LOCAL_CKPT}" --device cuda \
  --splits test_interp test_unseenN_interp test_unseenN test_unseenSigma
# evaluate writes under results/ml/<run_id>/
upload "results/ml/${LOCAL_RID}" "${BUCKET}/runs/${LOCAL_RID}/eval/"
upload "results/ml/runs/${LOCAL_RID}" "${BUCKET}/runs/${LOCAL_RID}/"
echo "EVAL_LOCAL_WALL_S=$(( $(date +%s)-T0 ))"

echo "=== STEP2 gns_baseline epoch_06 ==="
T0=$(date +%s)
retry_once eval_base python3 -m ml.evaluate \
  --config ml/configs/gns_baseline_gpu.yaml \
  --checkpoint "${BASE_CKPT}" --device cuda \
  --splits test_interp test_unseenN_interp test_unseenN test_unseenSigma
upload "results/ml/${BASE_RID}" "${BUCKET}/runs/${BASE_RID}/eval/"
upload "results/ml/runs/${BASE_RID}" "${BUCKET}/runs/${BASE_RID}/"
echo "EVAL_BASE_WALL_S=$(( $(date +%s)-T0 ))"

echo "STEP2_WALL_S=$(( $(date +%s)-T0_ALL ))"
upload "${LOGDIR}" "${BUCKET}/logs/table1b_final/"
echo "STEP1_STEP2_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "READY_FOR_STEP3"
