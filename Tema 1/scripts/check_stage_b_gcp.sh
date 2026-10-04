#!/usr/bin/env bash
# Quick status check for a detached Tema 1 Stage B GCP job.
#
# Usage:
#   bash "Tema 1/scripts/check_stage_b_gcp.sh"
#   INSTANCE=poliphysics-tema1-stageb ZONE=us-central1-a bash "Tema 1/scripts/check_stage_b_gcp.sh"
set -euo pipefail

INSTANCE="${INSTANCE:-poliphysics-tema1-stageb}"
ZONE="${ZONE:-us-central1-a}"
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
  echo '=== job.pid ==='
  if [[ -f ${REMOTE_DIR}/job.pid ]]; then
    PID=\$(cat ${REMOTE_DIR}/job.pid)
    echo PID=\$PID
    if ps -p \$PID -o pid=,etime=,cmd= 2>/dev/null; then
      echo STATE=RUNNING
    else
      echo STATE=NOT_RUNNING
    fi
  else
    echo 'no job.pid'
    echo STATE=UNKNOWN
  fi
  echo '=== DONE markers ==='
  for f in sweep_DONE cost_probe_DONE prep_DONE; do
    if [[ -f ${REMOTE_DIR}/results/\$f ]]; then
      echo \"\$f: PRESENT\"
      cat ${REMOTE_DIR}/results/\$f
    else
      echo \"\$f: absent\"
    fi
  done
  echo '=== stage_b.log (tail) ==='
  tail -n 25 ${REMOTE_DIR}/stage_b.log 2>/dev/null || echo '(no log yet)'
  echo '=== sweep_log.txt (tail) ==='
  tail -n 15 ${REMOTE_DIR}/results/sweep_log.txt 2>/dev/null || echo '(no sweep_log yet)'
  echo '=== sweep.csv rows ==='
  if [[ -f ${REMOTE_DIR}/results/sweep.csv ]]; then
    # header + data rows
    echo \$((\$(wc -l < ${REMOTE_DIR}/results/sweep.csv) - 1)) data rows
  else
    echo 'no sweep.csv'
  fi
  echo '=== ml_samples ==='
  du -sh ${REMOTE_DIR}/results/ml_samples 2>/dev/null || echo '(none)'
" || {
  echo "SSH failed (VM may still be booting)." >&2
  exit 2
}
