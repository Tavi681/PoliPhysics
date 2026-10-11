#!/usr/bin/env bash
# Runs on the GPU VM: patch norms (a_by_type; delete noise_stats), train configs sequentially.
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

# Ensure a_by_type in norm_stats; delete legacy noise_stats; refresh checksums (+GCS).
if python3 -c "import json; d=json.load(open('${DEST}/norm_stats.json')); raise SystemExit(0 if 'a_by_type' in d else 1)"; then
  echo "norm_stats already has a_by_type"
  # Still purge any leftover noise_stats locally + on GCS.
  rm -f "${DEST}/noise_stats.json" "${DEST}/noise_stats.sha256"
else
  echo "Patching norm_stats with a_by_type (full train Welford)..."
  python3 -m ml.scripts.patch_norm_a_by_type \
    --data-dir "${DEST}" \
    --upload-gcs "${BUCKET}/processed/${DATA_VERSION}/"
fi
# Ensure GCS noise_stats is gone even if local already patched.
gcloud storage rm "${BUCKET}/processed/${DATA_VERSION}/noise_stats.json" 2>/dev/null || true
gcloud storage rm "${BUCKET}/processed/${DATA_VERSION}/noise_stats.sha256" 2>/dev/null || true

# Gate accepted at ~68% train util / 2.6% val — skip re-gate for production trains.
export UTIL_GATE="${UTIL_GATE:-0}"
export UTIL_GATE_MIN
export UTIL_GATE_PCT
export UTIL_GATE_VAL_FRAC="${UTIL_GATE_VAL_FRAC:-0.10}"
# Session cost cap shared across sequential configs (wall clock from this script).
export TRAIN_SESSION_T0="${TRAIN_SESSION_T0:-$(date +%s)}"
export COST_CAP_USD="${COST_CAP_USD:-40}"
export VM_USD_PER_HOUR="${VM_USD_PER_HOUR:-0.853624312}"
python3 -m pip install -q nvidia-ml-py psutil 2>/dev/null || true
echo "SESSION_T0=${TRAIN_SESSION_T0} COST_CAP_USD=${COST_CAP_USD} VM_USD_PER_HOUR=${VM_USD_PER_HOUR}"

# W&B API key: prefer env already set at VM start; else GCP Secret Manager.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
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

for CFG in ${CONFIGS}; do
  echo "=== TRAIN ${CFG} @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  LOG="Tema 1/results/ml/logs/$(basename "${CFG}" .yaml)_train.log"
  python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
  python3 -m ml.train --config "${CFG}" --device cuda > "${LOG}" 2>&1 &
  TPID=$!
  tail -F "${LOG}" &
  TAILPID=$!
  (
    sleep "${HOUR_REPORT_S}"
    echo "=== 1h REPORT $(basename "${CFG}") $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv || true
    grep -E '^(STEP|NVML|UTIL|PHASE|wandb)' "${LOG}" | tail -n 40 || true
    RUN=$(ls -1dt "Tema 1/results/ml/runs"/*/ 2>/dev/null | head -1 || true)
    if [[ -n "${RUN}" && -f "${RUN}/best.pt" ]]; then
      echo "1h sanity diagnose on ${RUN}"
      SANITY_OUT="Tema 1/results/ml/logs/sanity_1h_$(basename "${CFG}" .yaml).json"
      if ! python3 -m ml.scripts.diagnose_loss_refs \
          --config "${CFG}" --checkpoint "${RUN}/best.pt" --device cuda \
          --out "${SANITY_OUT}" --n-batches 20 --require-below-zero-acc; then
        echo "SANITY_FAIL — killing train PID ${TPID}"
        kill "${TPID}" 2>/dev/null || true
        echo 44 > "Tema 1/results/ml/logs/sanity_fail.code"
      else
        echo "SANITY_OK — continuing"
      fi
    else
      echo "1h sanity skipped (no best.pt yet)"
    fi
    tail -n 40 "${LOG}" || true
  ) &
  SANITY_PID=$!
  set +e
  wait "${TPID}"
  RC=$?
  set -e
  kill "${TAILPID}" 2>/dev/null || true
  kill "${SANITY_PID}" 2>/dev/null || true
  if [[ -f "Tema 1/results/ml/logs/sanity_fail.code" ]]; then
    echo "Sanity check failed after 1h — stop session (VM stop left to launcher)."
    exit 44
  fi
  if [[ ${RC} -eq 42 ]]; then
    echo "GPU util gate failed — STOP."
    exit 42
  fi
  COST_HIT=0
  if [[ ${RC} -eq 43 ]]; then
    echo "COST_CAP hit — evaluate current run then stop session."
    COST_HIT=1
    RC=0
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
  if [[ ${COST_HIT} -eq 1 ]]; then
    echo "COST_CAP_SESSION_STOP"
    exit 43
  fi
done
echo DONE_TRAIN
