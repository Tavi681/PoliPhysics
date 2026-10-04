#!/usr/bin/env bash
# Status check for a detached Tema 1 Stage B GCP job (B0/R or B1/B2).
#
# Usage:
#   bash "Tema 1/scripts/check_stage_b_gcp.sh"
#   INSTANCE=poliphysics-tema1-b0 bash "Tema 1/scripts/check_stage_b_gcp.sh"
#   INSTANCE=poliphysics-tema1-b12-s0 bash "Tema 1/scripts/check_stage_b_gcp.sh"
set -euo pipefail

INSTANCE="${INSTANCE:-poliphysics-tema1-b0}"
ZONE="${ZONE:-us-central1-a}"
REMOTE_BASE="${REMOTE_BASE:-/mnt/pd/tema1}"
REMOTE_NEW="${REMOTE_NEW:-${REMOTE_BASE}/new}"
# Fallback for the older home-directory launcher.
REMOTE_DIR="${REMOTE_DIR:-/home/octav/tema1}"

export CLOUDSDK_CORE_DISABLE_FILE_LOGGING=1
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

if ! gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" &>/dev/null; then
  echo "INSTANCE ${INSTANCE} not found in ${ZONE} (deleted or wrong name/zone)."
  exit 1
fi

STATUS="$(gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" --format='get(status)')"
echo "VM ${INSTANCE} (${ZONE}): ${STATUS}"

gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set +e
  echo '=== provenance ==='
  echo -n 'COMMIT='; tr -d '[:space:]' < ${REMOTE_NEW}/COMMIT 2>/dev/null; echo
  echo -n 'COMMIT.dirty='; tr -d '[:space:]' < ${REMOTE_NEW}/COMMIT.dirty 2>/dev/null; echo
  cat ${REMOTE_BASE}/deploy_gate.txt 2>/dev/null
  echo '=== disks / mount ==='
  df -h /mnt/pd 2>/dev/null || echo '(no /mnt/pd)'
  echo '=== tmux ==='
  tmux list-sessions 2>/dev/null || echo '(no tmux)'
  echo '=== pids ==='
  for f in ${REMOTE_BASE}/b0.pid ${REMOTE_BASE}/r.pid ${REMOTE_BASE}/rsync.pid ${REMOTE_DIR}/job.pid; do
    if [[ -f \$f ]]; then
      PID=\$(cat \$f)
      echo \$f PID=\$PID
      ps -p \$PID -o pid=,etime=,cmd= 2>/dev/null || echo '  NOT_RUNNING'
    fi
  done
  echo '=== DONE markers ==='
  for base in ${REMOTE_NEW}/results ${REMOTE_DIR}/results; do
    [[ -d \$base ]] || continue
    echo \"dir \$base\"
    for f in B0_DONE R_DONE sweep_DONE cost_probe_DONE prep_DONE sweep_DONE_shard_00 sweep_DONE_shard_01 sweep_DONE_shard_02; do
      if [[ -f \$base/\$f ]]; then
        echo \"  \$f: PRESENT\"
        cat \$base/\$f
      fi
    done
  done
  echo '=== reports (head) ==='
  for f in ${REMOTE_NEW}/results/regression.txt ${REMOTE_NEW}/results/b0_report.txt ${REMOTE_NEW}/results/b0_estimate.txt; do
    if [[ -f \$f ]]; then
      echo \"--- \$f ---\"
      tail -n 20 \$f
    fi
  done
  echo '=== logs (tail) ==='
  tail -n 15 ${REMOTE_NEW}/results/b0.log 2>/dev/null
  tail -n 10 ${REMOTE_NEW}/results/r.log 2>/dev/null
  tail -n 15 ${REMOTE_DIR}/stage_b.log 2>/dev/null
  tail -n 10 ${REMOTE_NEW}/results/sweep_log.txt 2>/dev/null
  tail -n 10 ${REMOTE_DIR}/results/sweep_log.txt 2>/dev/null
  echo '=== sweep.csv rows ==='
  for f in ${REMOTE_NEW}/results/sweep.csv ${REMOTE_NEW}/results/sweep_shard_00.csv ${REMOTE_DIR}/results/sweep.csv; do
    if [[ -f \$f ]]; then
      echo \$f: \$((\$(wc -l < \$f) - 1)) data rows
    fi
  done
  echo '=== ml_samples ==='
  du -sh ${REMOTE_NEW}/results/ml_samples ${REMOTE_DIR}/results/ml_samples 2>/dev/null || echo '(none)'
" || {
  echo "SSH failed (VM may still be booting)." >&2
  exit 2
}
