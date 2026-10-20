#!/usr/bin/env bash
# Long poll VM baseline train + post pipeline (do NOT kill ml.train).
set -euo pipefail
INSTANCE=poliphysics-tema1-ml-train
ZONE=us-central1-b
RID=gns_baseline_s42_5716ee5b_1791828699
R="/home/octav/poliphysics/Tema 1/results/ml"
OUT="/Users/octav/Projects/PoliPhysics/PoliPhysics/Tema 1/results/ml/logs/watch_baseline_long_poll.log"
POLL_S="${POLL_S:-720}"  # 12 min default

ssh_vm() {
  gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="$1" 2>&1
}

poll_once() {
  local utc
  utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  ssh_vm "
    LOG='${R}/logs/gns_baseline_gpu_train.log'
    RUN='${R}/runs/${RID}'
    STEP=\$(grep -E '^STEP ' \"\${LOG}\" 2>/dev/null | tail -1 || true)
    EPOCH_CKPT=\$(ls \"\${RUN}\"/epoch_*.pt 2>/dev/null | wc -l | tr -d ' ')
    TRAIN=\$(pgrep -af 'python3 -m ml.train' 2>/dev/null | grep -v grep | head -1 || echo 'DEAD')
    POST=\$(pgrep -af post_baseline_pipeline 2>/dev/null | grep -v grep | head -1 || echo 'no')
    WAITER=\$(tail -1 '${R}/logs/post_waiter.log' 2>/dev/null || echo '')
    VM_STATUS=\$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/status 2>/dev/null || echo '?')
    echo 'POLL utc=${utc}'
    echo \"STEP_LINE \${STEP}\"
    echo \"EPOCH_PT_COUNT \${EPOCH_CKPT}\"
    echo \"TRAIN \${TRAIN}\"
    echo \"POST \${POST}\"
    echo \"WAITER \${WAITER}\"
    echo \"VM \${VM_STATUS}\"
    ls \"\${RUN}\"/epoch_*.pt 2>/dev/null || true
    tail -3 '${R}/logs/post_baseline_pipeline.log' 2>/dev/null || true
  "
}

echo "=== watch start $(date -u +%Y-%m-%dT%H:%M:%SZ) RID=${RID} poll=${POLL_S}s ===" | tee -a "${OUT}"

while true; do
  {
    echo "---"
    poll_once
  } | tee -a "${OUT}"

  # Done: VM stopped and post succeeded (or poll sees post rc=0 in waiter log)
  if grep -q 'post pipeline rc=0' "${OUT}" 2>/dev/null; then
    echo "DONE post rc=0 $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT}"
    exit 0
  fi
  W=$(ssh_vm "cat '${R}/logs/post_waiter.log' 2>/dev/null; true" || true)
  if echo "${W}" | grep -q 'post pipeline rc=0'; then
    echo "DONE waiter post rc=0 $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT}"
    exit 0
  fi
  if echo "${W}" | grep -qE 'post pipeline rc=[1-9]'; then
    echo "FAIL post pipeline $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT}"
    exit 2
  fi
  # VM stopped without post success — keep logging but may need manual check
  ST=$(gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" --format='get(status)' 2>/dev/null || echo UNKNOWN)
  if [[ "${ST}" == "TERMINATED" ]]; then
    echo "VM TERMINATED $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT}"
    if echo "${W}" | grep -q 'post pipeline rc=0'; then
      exit 0
    fi
    exit 3
  fi

  sleep "${POLL_S}"
done
