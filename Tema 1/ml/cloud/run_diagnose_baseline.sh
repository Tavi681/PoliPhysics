#!/usr/bin/env bash
# gns_baseline failure diagnostics (edge drift, spider z, one-step pre-fail).
# Works on CPU or GPU; full val sample needs processed 5c60fa669a719404 on disk.
#
# On CPU VM (after ssh, repo at ~/poliphysics, data at /data/processed/5c60fa669a719404):
#   cd ~/poliphysics/Tema\ 1 && export PYTHONPATH=.
#
#   bash ml/cloud/run_diagnose_baseline.sh cpu
#
# GPU (optional, faster model forward):
#   bash ml/cloud/run_diagnose_baseline.sh cuda
#
# Pull selected checkpoint from GCS if missing locally:
#   BUCKET=gs://authorship-verification-poliphysics/tema1-ml
#   RID=gns_baseline_s42_5716ee5b_1791828699
#   gsutil cp "${BUCKET}/runs/${RID}/ckpt/epoch_04.pt" "results/ml/runs/${RID}/"
#
set -euo pipefail

DEVICE="${1:-cpu}"
if [[ -n "${REMOTE_DIR:-}" ]]; then
  ROOT="${REMOTE_DIR}/Tema 1"
else
  ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi
cd "${ROOT}"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

CKPT="${CKPT:-results/ml/runs/gns_baseline_s42_5716ee5b_1791828699/epoch_04.pt}"
OUT="${OUT:-results/ml/diagnostics/gns_baseline_epoch04_val20}"
N_RUNS="${N_RUNS:-20}"
T_END="${T_END:-60}"

python3 -m ml.scripts.diagnose_baseline_failure \
  --config ml/configs/gns_baseline_gpu.yaml \
  --checkpoint "${CKPT}" \
  --device "${DEVICE}" \
  --n-runs "${N_RUNS}" \
  --t-end "${T_END}" \
  --out-dir "${OUT}"

echo "Diagnostics written under ${OUT}/ (summary.json, summary.md, PNGs)"
