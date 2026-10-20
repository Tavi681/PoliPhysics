#!/usr/bin/env bash
# Robust long watch: survives SSH blips; poll 25–40 min; never kill ml.train.
set -uo pipefail
INSTANCE=poliphysics-tema1-ml-train
ZONE=us-central1-b
RID=gns_baseline_s42_5716ee5b_1791828699
R="/home/octav/poliphysics/Tema 1/results/ml"
OUT="/Users/octav/Projects/PoliPhysics/PoliPhysics/Tema 1/results/ml/logs/agent_long_watch_resume.log"
DONE="/Users/octav/Projects/PoliPhysics/PoliPhysics/Tema 1/results/ml/logs/agent_long_watch.DONE"

ssh_vm() {
  local cmd="$1" i out=""
  for i in 1 2 3 4 5; do
    if out=$(gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="$cmd" 2>&1); then
      printf '%s\n' "$out"
      return 0
    fi
    echo "SSH_RETRY ${i} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT}"
    sleep $((15 * i))
  done
  return 1
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
  ssh_vm "cd /home/octav/poliphysics && nohup bash ml/cloud/wait_then_post.sh >> '${R}/logs/wait_then_post.nohup' 2>&1 & sleep 2; pgrep -af wait_then_post || true" || true
}

start_post_nohup() {
  echo "ACTION start post_baseline_pipeline nohup $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT}"
  ssh_vm "cd /home/octav/poliphysics && export BASE_RID=${RID} && nohup bash ml/cloud/post_baseline_pipeline.sh >> '${R}/logs/post_baseline_pipeline.log' 2>&1 & sleep 2; pgrep -af post_baseline_pipeline || true" || true
}

maybe_fix_post() {
  echo "ACTION diagnose post failure $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT}"
  ssh_vm "tail -80 '${R}/logs/post_baseline_pipeline.log' 2>/dev/null" | tee -a "${OUT}" || true
  # If post died mid-pipeline and no post process, re-run full pipeline (idempotent eval dirs).
  if ! ssh_vm "pgrep -af post_baseline_pipeline | grep -v grep" >/dev/null 2>&1; then
    start_post_nohup
  fi
}

rand_sleep() {
  local lo=1500 hi=2400 span=$((hi - lo))
  sleep $((lo + RANDOM % (span + 1)))
}

echo "=== agent_long_watch_resume start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "${OUT}"
rm -f "${DONE}"

while true; do
  {
    echo "--- POLL $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
    poll_vm || echo "SSH_POLL_FAILED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } | tee -a "${OUT}"

  W=$(ssh_vm "cat '${R}/logs/post_waiter.log' 2>/dev/null; true" 2>/dev/null || true)
  if echo "${W}" | grep -q 'post pipeline rc=0'; then
    ST=$(gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" --format='get(status)' 2>/dev/null || echo UNKNOWN)
    echo "DONE post rc=0 VM=${ST} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT}"
    echo "post rc=0 VM=${ST}" > "${DONE}"
    exit 0
  fi
  if echo "${W}" | grep -qE 'post pipeline rc=[1-9]'; then
    maybe_fix_post
  fi

  VM_BLOCK=$(poll_vm 2>/dev/null || echo "SSH_POLL_FAILED")
  TRAIN_CNT=$(echo "${VM_BLOCK}" | grep '^TRAIN_CNT ' | awk '{print $2}' || echo 0)
  WAITER_LINE=$(echo "${VM_BLOCK}" | grep '^WAITER ' | sed 's/^WAITER //' || true)
  POST_LINE=$(echo "${VM_BLOCK}" | grep '^POST ' | sed 's/^POST //' || true)

  if [[ "${TRAIN_CNT}" -gt 0 ]] && [[ -z "${WAITER_LINE}" ]]; then
    restart_waiter
  fi

  if [[ "${TRAIN_CNT}" -eq 0 ]] && [[ -z "${POST_LINE}" ]]; then
    if ! echo "${W}" | grep -q 'starting post pipeline'; then
      LAST_MOD=$(ssh_vm "date +%s; stat -c %Y '${R}/logs/gns_baseline_gpu_train.log' 2>/dev/null || echo 0" 2>/dev/null || echo "0 0")
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
      echo "post rc=0 VM=TERMINATED" > "${DONE}"
      exit 0
    fi
    exit 3
  fi
  if [[ "${ST}" == "STOPPED" ]] && echo "${W}" | grep -q 'post pipeline rc=0'; then
    echo "DONE post rc=0 VM=STOPPED $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT}"
    echo "post rc=0 VM=STOPPED" > "${DONE}"
    exit 0
  fi

  rand_sleep
done
