#!/usr/bin/env bash
# Runs on the GPU VM: ensure local data + noise_stats, train configs sequentially.
# Env: DATA_VERSION, BUCKET, CONFIGS, UTIL_GATE_MIN, UTIL_GATE_PCT, REMOTE_DIR
set -euo pipefail
cd "${REMOTE_DIR:-/home/octav/poliphysics}"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-poliphysics-ballooning-ml}"
export DATA_VERSION="${DATA_VERSION:-5c60fa669a719404}"

BUCKET="${BUCKET:-gs://authorship-verification-poliphysics/tema1-ml}"
DEST="/data/processed/${DATA_VERSION}"
CONFIGS="${CONFIGS:-ml/configs/gns_local_gpu.yaml ml/configs/gns_baseline_gpu.yaml}"
UTIL_GATE_MIN="${UTIL_GATE_MIN:-15}"
UTIL_GATE_PCT="${UTIL_GATE_PCT:-70}"
HOUR_REPORT_S="${HOUR_REPORT_S:-3600}"

mkdir -p "${DEST}" "Tema 1/results/ml/runs" "Tema 1/results/ml/logs"
if [[ ! -f "${DEST}/DATA_VERSION" ]]; then
  echo "Syncing ${BUCKET}/processed/${DATA_VERSION} → ${DEST}"
  gcloud storage rsync --recursive "${BUCKET}/processed/${DATA_VERSION}" "${DEST}" \
    || gsutil -m rsync -r "${BUCKET}/processed/${DATA_VERSION}" "${DEST}"
fi

python3 -m pip install -q pandas pyyaml wandb 2>/dev/null || true

# Full train-split noise_stats once (checksummed); never recompute if present.
if [[ ! -f "${DEST}/noise_stats.json" || ! -f "${DEST}/noise_stats.sha256" ]]; then
  echo "Computing full-train noise_stats (one-time)..."
  python3 -m ml.scripts.compute_noise_stats \
    --data-dir "${DEST}" \
    --upload-gcs "${BUCKET}/processed/${DATA_VERSION}/"
else
  echo "noise_stats present with checksum — skipping recompute"
  cat "${DEST}/noise_stats.sha256"
fi

# UTIL_GATE is enforced inside ml.train (NVML 1 Hz from first training step;
# train-only util + val wall-fraction ≤ 10%).
export UTIL_GATE=1
export UTIL_GATE_MIN
export UTIL_GATE_PCT
export UTIL_GATE_VAL_FRAC="${UTIL_GATE_VAL_FRAC:-0.10}"
python3 -m pip install -q nvidia-ml-py psutil 2>/dev/null || true

# W&B API key: prefer env already set at VM start; else GCP Secret Manager.
# Never write the key into the repo or into logs.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_SECRET="${WANDB_SECRET:-projects/authorship-verification/secrets/wandb-api-key/versions/latest}"
  if WANDB_API_KEY="$(gcloud secrets versions access latest --secret=wandb-api-key 2>/dev/null)"; then
    export WANDB_API_KEY
    echo "W&B: loaded API key from Secret Manager (wandb-api-key)"
  else
    echo "W&B: WANDB_API_KEY unset and Secret Manager unavailable — wandb may go offline"
  fi
else
  echo "W&B: using WANDB_API_KEY from environment"
fi
export WANDB_MODE="${WANDB_MODE:-online}"
unset WANDB_SECRET

for CFG in ${CONFIGS}; do
  echo "=== TRAIN ${CFG} @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  LOG="Tema 1/results/ml/logs/$(basename "${CFG}" .yaml)_train.log"
  # Fail loud if CUDA missing
  python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
  # Redirect (not pipe) so wait sees ml.train's exit code (42 = util gate fail).
  python3 -m ml.train --config "${CFG}" --device cuda > "${LOG}" 2>&1 &
  TPID=$!
  tail -F "${LOG}" &
  TAILPID=$!
  (
    sleep "${HOUR_REPORT_S}"
    echo "=== 1h REPORT $(basename "${CFG}") $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv || true
    grep -E '^(STEP|NVML|UTIL|PHASE)' "${LOG}" | tail -n 40 || true
    tail -n 40 "${LOG}" || true
  ) &
  set +e
  wait "${TPID}"
  RC=$?
  set -e
  kill "${TAILPID}" 2>/dev/null || true
  if [[ ${RC} -eq 42 ]]; then
    echo "GPU util gate failed — STOP. Stop (not delete) VM after report."
    exit 42
  fi
  if [[ ${RC} -ne 0 ]]; then
    echo "train exited rc=${RC}"
    exit "${RC}"
  fi
  RUN=$(ls -1dt "Tema 1/results/ml/runs"/*/ | head -1)
  echo "Evaluating ${RUN}"
  python3 -m ml.evaluate --config "${CFG}" --checkpoint "${RUN}/best.pt" --device cuda \
    --splits test_interp test_unseenN_interp test_unseenN test_unseenSigma
  RID=$(basename "${RUN}")
  gsutil -m rsync -r "${RUN}" "${BUCKET}/runs/${RID}/" || true
done
echo DONE_TRAIN
