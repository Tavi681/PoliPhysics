#!/usr/bin/env bash
# Agent long watch: poll VM baseline train → post → VM stop (never kill ml.train).
set -euo pipefail
INSTANCE=poliphysics-tema1-ml-train
ZONE=us-central1-b
RID=gns_baseline_s42_5716ee5b_1791828699
R="/home/octav/poliphysics/Tema 1/results/ml"
OUT="/Users/octav/Projects/PoliPhysics/PoliPhysics/Tema 1/results/ml/logs/agent_long_watch.log"
POLL_S="${POLL_S:-900}"

ssh_vm() {
  gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="$1" 2>&1
}

poll_vm() {
  ssh_vm "
    LOG='${R}/logs/gns_baseline_gpu_train.log'
    RUN='${R}/runs/${RID}'
    STEP=\$(grep -E '^STEP ' \"\${LOG}\" 2>/dev/null | tail -1 || true)
    EPOCH_CKPT=\$(ls \"\${RUN}\"/epoch_*.pt 2>/dev/null | wc -l | tr -d ' ')
    TRAIN_CNT=\$(pgrep -fc 'python3 -m ml.train' 2>/dev/null || echo 0)
    WAITER=\$(pgrep -af 'wait_then_post' 2>/dev/null | grep -v grep | head -1 || echo '')
    POST=\$(pgrep -af 'post_baseline_pipeline' 2>/dev/null | grep -v grep | head -1 || echo '')
    echo \"STEP \${STEP}\"
    echo \"EPOCH_PT \${EPOCH_CKPT}\"
    echo \"TRAIN_CNT \${TRAIN_CNT}\"
    echo \"WAITER \${WAITER}\"
    echo \"POST \${POST}\"
    ls \"\${RUN}\"/epoch_*.pt 2>/dev/null || true
    tail -1 '${R}/logs/post_waiter.log' 2>/dev/null || true
    tail -2 '${R}/logs/post_baseline_pipeline.log' 2>/dev/null || true
  "
}

restart_waiter() {
  echo "ACTION restart wait_then_post $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT}"
  ssh_vm "cd /home/octav/poliphysics && nohup bash ml/cloud/wait_then_post.sh >> '${R}/logs/wait_then_post.nohup' 2>&1 & sleep 2; pgrep -af wait_then_post || true"
}

start_post_nohup() {
  echo "ACTION start post_baseline_pipeline nohup $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT}"
  ssh_vm "cd /home/octav/poliphysics && export BASE_RID=${RID} && nohup bash ml/cloud/post_baseline_pipeline.sh >> '${R}/logs/post_baseline_pipeline.log' 2>&1 & sleep 2; pgrep -af post_baseline_pipeline || true"
}

echo "=== agent_long_watch start $(date -u +%Y-%m-%dT%H:%M:%SZ) poll=${POLL_S}s ===" | tee -a "${OUT}"

while true; do
  {
    echo "--- POLL $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
    poll_vm || echo "SSH_POLL_FAILED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } | tee -a "${OUT}"

  W=$(ssh_vm "cat '${R}/logs/post_waiter.log' 2>/dev/null; true" || true)
  if echo "${W}" | grep -q 'post pipeline rc=0'; then
    echo "DONE post rc=0 $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT}"
    ST=$(gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" --format='get(status)' 2>/dev/null || echo UNKNOWN)
    echo "VM_STATUS ${ST}" | tee -a "${OUT}"
    exit 0
  fi
  if echo "${W}" | grep -qE 'post pipeline rc=[1-9]'; then
    echo "WARN post failed — see post_baseline_pipeline.log $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT}"
    # Do not exit; agent may fix manually on next poll
  fi

  VM_BLOCK=$(poll_vm 2>/dev/null || echo "SSH_POLL_FAILED")
  TRAIN_CNT=$(echo "${VM_BLOCK}" | grep '^TRAIN_CNT ' | awk '{print $2}' || echo 0)
  WAITER_LINE=$(echo "${VM_BLOCK}" | grep '^WAITER ' | sed 's/^WAITER //' || true)
  POST_LINE=$(echo "${VM_BLOCK}" | grep '^POST ' | sed 's/^POST //' || true)

  if [[ "${TRAIN_CNT}" -gt 0 ]] && [[ -z "${WAITER_LINE}" ]]; then
    restart_waiter
  fi

  if [[ "${TRAIN_CNT}" -eq 0 ]] && [[ -z "${POST_LINE}" ]]; then
    GONE=$(ssh_vm "grep -E 'ml.train gone|stop marker|starting post' '${R}/logs/post_waiter.log' 2>/dev/null | tail -1; stat -c %Y '${R}/logs/gns_baseline_gpu_train.log' 2>/dev/null || echo 0" || true)
    # If train dead and no post process, check log mtime / waiter
    if ! echo "${W}" | grep -q 'starting post pipeline'; then
      LAST_MOD=$(ssh_vm "date +%s; stat -c %Y '${R}/logs/gns_baseline_gpu_train.log' 2>/dev/null || echo 0" || echo "0 0")
      NOW=$(echo "${LAST_MOD}" | head -1)
      LM=$(echo "${LAST_MOD}" | tail -1)
      if [[ "${NOW}" -gt 0 && "${LM}" -gt 0 ]] && (( NOW - LM >= 120 )); then
        start_post_nohup
      fi
    fi
  fi

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
