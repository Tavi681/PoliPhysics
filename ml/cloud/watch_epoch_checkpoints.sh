#!/usr/bin/env bash
# Watch a running train log; when epoch increments, copy last.pt → epoch_XX.pt
# and sync that file to GCS. Does NOT touch the training process.
#
# Usage (on VM):
#   bash ml/cloud/watch_epoch_checkpoints.sh \
#     --run-dir "Tema 1/results/ml/runs/gns_baseline_s42_5716ee5b_1791828699" \
#     --log "Tema 1/results/ml/logs/gns_baseline_gpu_train.log"
set -euo pipefail
RUN_DIR=""
LOG=""
BUCKET="${BUCKET:-gs://authorship-verification-poliphysics/tema1-ml}"
POLL_S="${POLL_S:-15}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --poll) POLL_S="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done
[[ -n "${RUN_DIR}" && -n "${LOG}" ]] || { echo "need --run-dir and --log"; exit 2; }

cd "${REMOTE_DIR:-/home/octav/poliphysics}"
RID=$(basename "${RUN_DIR}")
STATE="Tema 1/results/ml/logs/epoch_watch_${RID}.state"
mkdir -p "$(dirname "${STATE}")"
LAST_SEEN_EPOCH=-1
if [[ -f "${STATE}" ]]; then
  LAST_SEEN_EPOCH=$(cat "${STATE}" || echo -1)
fi
echo "epoch_watch start RID=${RID} last_seen=${LAST_SEEN_EPOCH} $(date -u +%Y-%m-%dT%H:%M:%SZ)"

copy_epoch() {
  local ep="$1"
  local src="${RUN_DIR}/last.pt"
  local dst
  printf -v dst "%s/epoch_%02d.pt" "${RUN_DIR}" "${ep}"
  if [[ ! -f "${src}" ]]; then
    echo "epoch_watch: no last.pt yet for epoch ${ep}"
    return 0
  fi
  cp -f "${src}" "${dst}"
  echo "epoch_watch: wrote ${dst} from last.pt @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  gsutil -q cp "${dst}" "${BUCKET}/runs/${RID}/epoch_$(printf '%02d' "${ep}").pt" \
    && echo "epoch_watch: synced GCS epoch_$(printf '%02d' "${ep}").pt" \
    || echo "epoch_watch: GCS sync failed (will retry next poll)"
}

# last.pt is updated every val_every steps (not only at epoch end). When the log
# first shows epoch=N (N>=1), epoch N-1 has just completed — snapshot last.pt.
while true; do
  if [[ ! -f "${LOG}" ]]; then
    sleep "${POLL_S}"
    continue
  fi
  # Highest epoch mentioned in recent STEP lines
  CUR=$(grep -E '^STEP ' "${LOG}" | tail -n 200 | sed -n 's/.* epoch=\([0-9][0-9]*\).*/\1/p' | tail -n 1 || true)
  if [[ -z "${CUR}" ]]; then
    sleep "${POLL_S}"
    continue
  fi
  if [[ "${CUR}" -gt "${LAST_SEEN_EPOCH}" ]]; then
    # For each newly observed epoch value E>=1, snapshot completed epoch E-1.
    for ((e=LAST_SEEN_EPOCH+1; e<=CUR; e++)); do
      if [[ "${e}" -ge 1 ]]; then
        copy_epoch $((e - 1))
      fi
    done
    LAST_SEEN_EPOCH="${CUR}"
    echo "${LAST_SEEN_EPOCH}" > "${STATE}"
  fi
  # Terminal conditions
  if grep -qE 'EPOCH STOP:|EARLY_STOP:|TRAIN_DONE|COST_CAP STOP' "${LOG}" 2>/dev/null; then
    # Final snapshot for the last completed epoch index still showing in STEP
    FINAL=$(grep -E '^STEP ' "${LOG}" | sed -n 's/.* epoch=\([0-9][0-9]*\).*/\1/p' | tail -n 1 || true)
    if [[ -n "${FINAL}" ]]; then
      copy_epoch "${FINAL}"
    fi
    echo "epoch_watch: train log ended — exiting $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 0
  fi
  # Also exit if train process gone and log idle > 2 min
  if ! pgrep -af 'ml.train' >/dev/null 2>&1; then
    sleep 30
    if ! pgrep -af 'ml.train' >/dev/null 2>&1; then
      FINAL=$(grep -E '^STEP ' "${LOG}" | sed -n 's/.* epoch=\([0-9][0-9]*\).*/\1/p' | tail -n 1 || true)
      if [[ -n "${FINAL}" ]]; then
        copy_epoch "${FINAL}"
      fi
      echo "epoch_watch: no ml.train — exiting"
      exit 0
    fi
  fi
  sleep "${POLL_S}"
done
