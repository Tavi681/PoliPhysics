#!/usr/bin/env bash
# Runs on the GPU VM: patch norms (a_by_type; delete noise_stats), train configs sequentially.
# Env: DATA_VERSION, BUCKET, CONFIGS, UTIL_GATE_MIN, UTIL_GATE_PCT, REMOTE_DIR,
#      COST_CAP_USD, VM_USD_PER_HOUR, TRAIN_SESSION_T0, HOUR_REPORT_S
set -euo pipefail
cd "${REMOTE_DIR:-/home/octav/poliphysics}/Tema 1"
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
SANITY_TIMEOUT_S="${SANITY_TIMEOUT_S:-1800}"

mkdir -p "${DEST}" "results/ml/runs" "results/ml/logs"
if [[ ! -f "${DEST}/DATA_VERSION" ]]; then
  echo "Syncing ${BUCKET}/processed/${DATA_VERSION} → ${DEST}"
  gcloud storage rsync --recursive "${BUCKET}/processed/${DATA_VERSION}" "${DEST}" \
    || gsutil -m rsync -r "${BUCKET}/processed/${DATA_VERSION}" "${DEST}"
fi

python3 -m pip install -q pandas pyyaml wandb 2>/dev/null || true

# Ensure a_by_type in norm_stats; delete legacy noise_stats; refresh checksums (+GCS).
if python3 -c "import json; d=json.load(open('${DEST}/norm_stats.json')); raise SystemExit(0 if 'a_by_type' in d else 1)"; then
  echo "norm_stats already has a_by_type"
  rm -f "${DEST}/noise_stats.json" "${DEST}/noise_stats.sha256"
else
  echo "Patching norm_stats with a_by_type (full train Welford)..."
  python3 -m ml.scripts.patch_norm_a_by_type \
    --data-dir "${DEST}" \
    --upload-gcs "${BUCKET}/processed/${DATA_VERSION}/"
fi
gcloud storage rm "${BUCKET}/processed/${DATA_VERSION}/noise_stats.json" 2>/dev/null || true
gcloud storage rm "${BUCKET}/processed/${DATA_VERSION}/noise_stats.sha256" 2>/dev/null || true

export UTIL_GATE="${UTIL_GATE:-0}"
export UTIL_GATE_MIN
export UTIL_GATE_PCT
export UTIL_GATE_VAL_FRAC="${UTIL_GATE_VAL_FRAC:-0.10}"
export TRAIN_SESSION_T0="${TRAIN_SESSION_T0:-$(date +%s)}"
export COST_CAP_USD="${COST_CAP_USD:-40}"
export VM_USD_PER_HOUR="${VM_USD_PER_HOUR:-0.853624312}"
python3 -m pip install -q nvidia-ml-py psutil 2>/dev/null || true
echo "SESSION_T0=${TRAIN_SESSION_T0} COST_CAP_USD=${COST_CAP_USD} VM_USD_PER_HOUR=${VM_USD_PER_HOUR}"

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

run_cpu_sanity() {
  # Never touch the training GPU. Copy ckpt; force empty CUDA_VISIBLE_DEVICES.
  local CFG="$1" RUN="$2" OUT="$3"
  local SRC=0
  echo "1h CPU sanity (CUDA_VISIBLE_DEVICES='') on copy of ${RUN}/best.pt"
  set +e
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=KILL "${SANITY_TIMEOUT_S}" \
      env CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
      python3 -m ml.scripts.sanity_check_cpu \
        --config "${CFG}" --checkpoint "${RUN}/best.pt" \
        --out "${OUT}" --n-batches 20
    SRC=$?
  else
    env CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
      python3 -m ml.scripts.sanity_check_cpu \
        --config "${CFG}" --checkpoint "${RUN}/best.pt" \
        --out "${OUT}" --n-batches 20
    SRC=$?
  fi
  set -e
  # timeout returns 137 on KILL
  if [[ ${SRC} -eq 44 ]]; then
    echo "SANITY_FAIL (metric) — will stop training"
    return 44
  fi
  if [[ ${SRC} -ne 0 ]]; then
    echo "SANITY_ERROR rc=${SRC} (exception/OOM/timeout) — training CONTINUES"
    echo "${SRC}" > "results/ml/logs/sanity_error.code"
    echo "SANITY_ERROR rc=${SRC} $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >> "results/ml/logs/sanity_error.log"
    return 0
  fi
  echo "SANITY_OK — continuing"
  return 0
}

sync_run_artifacts() {
  local RUN="$1"
  local RID
  RID=$(basename "${RUN}")
  local EVAL_DIR="results/ml/${RID}"
  # evaluate.py writes metrics next to results/ml/<run_id>/ — copy into runs/ for one GCS sync.
  if [[ -d "${EVAL_DIR}" ]]; then
    cp -f "${EVAL_DIR}/metrics.json" "${RUN}/" 2>/dev/null || true
    cp -f "${EVAL_DIR}/tab_accuracy.csv" "${RUN}/" 2>/dev/null || true
    cp -f "${EVAL_DIR}/tab_cost.csv" "${RUN}/" 2>/dev/null || true
  fi
  gsutil -m rsync -r "${RUN}" "${BUCKET}/runs/${RID}/" || true
  if [[ -d "${EVAL_DIR}" ]]; then
    gsutil -m rsync -r "${EVAL_DIR}" "${BUCKET}/runs/${RID}/eval/" || true
  fi
}

for CFG in ${CONFIGS}; do
  echo "=== TRAIN ${CFG} @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  LOG="results/ml/logs/$(basename "${CFG}" .yaml)_train.log"
  rm -f "results/ml/logs/sanity_fail.code"
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
    RUN=$(ls -1dt "results/ml/runs"/*/ 2>/dev/null | head -1 || true)
    if [[ -n "${RUN}" && -f "${RUN}/best.pt" ]]; then
      SANITY_OUT="results/ml/logs/sanity_1h_$(basename "${CFG}" .yaml).json"
      if ! run_cpu_sanity "${CFG}" "${RUN}" "${SANITY_OUT}"; then
        echo "SANITY_FAIL — killing train PID ${TPID}"
        kill "${TPID}" 2>/dev/null || true
        echo 44 > "results/ml/logs/sanity_fail.code"
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
  if [[ -f "results/ml/logs/sanity_fail.code" ]]; then
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
  RUN=$(ls -1dt "results/ml/runs"/*/ | head -1)
  echo "Evaluating ${RUN}"
  python3 -m ml.evaluate --config "${CFG}" --checkpoint "${RUN}/best.pt" --device cuda \
    --splits test_interp test_unseenN_interp test_unseenN test_unseenSigma
  sync_run_artifacts "${RUN}"
  if [[ ${COST_HIT} -eq 1 ]]; then
    echo "COST_CAP_SESSION_STOP"
    exit 43
  fi
done
echo DONE_TRAIN
