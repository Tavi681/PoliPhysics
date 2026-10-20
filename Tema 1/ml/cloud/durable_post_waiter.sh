#!/usr/bin/env bash
# VM-side fallback: survives Cursor/agent death. Polls every 60s.
# - While training: keep wait_then_post alive (restart if dead).
# - After train exit (+ marker or 60s gone): run post_baseline_pipeline once (flock).
set -euo pipefail
cd "/home/octav/poliphysics/Tema 1"
export BASE_RID="${BASE_RID:-gns_baseline_s42_5716ee5b_1791828699}"
LOG="results/ml/logs/gns_baseline_gpu_train.log"
OUT="results/ml/logs/post_baseline_pipeline.log"
W="results/ml/logs/durable_post_waiter.log"
LOCK="results/ml/logs/post_baseline_pipeline.started"
FLOCK="/var/tmp/poliphysics_post_pipeline.lock"
WAITER_FLOCK="/var/tmp/poliphysics_durable_waiter.lock"
STOP_RE='TRAIN_DONE|EARLY_STOP:|EPOCH STOP:|COST_CAP STOP|stop_reason='
TRAIN_GONE_SINCE=""

log() { echo "$(date -u +%FT%TZ) $*" >> "$W"; }

exec 8>"${WAITER_FLOCK}"
if ! flock -n 8; then
  log "another durable_post_waiter holds lock; exit"
  exit 0
fi

log "durable_post_waiter start pid=$$ BASE_RID=$BASE_RID"

while true; do
  if [[ -f "$OUT" ]] && grep -q 'DONE_POST_PIPELINE' "$OUT" 2>/dev/null; then
    log "DONE_POST_PIPELINE present; exiting"
    exit 0
  fi

  if pgrep -f 'post_baseline_pipeline\.sh' >/dev/null 2>&1; then
    log "post_baseline_pipeline running; sleep 60"
    sleep 60
    continue
  fi

  if pgrep -f 'python3 -m ml\.train' >/dev/null 2>&1; then
    TRAIN_GONE_SINCE=""
    if ! pgrep -f 'wait_then_post\.sh' >/dev/null 2>&1; then
      log "train alive; wait_then_post dead — restarting"
      nohup bash ml/cloud/wait_then_post.sh >> "results/ml/logs/wait_then_post.nohup" 2>&1 &
      sleep 2
    fi
    sleep 60
    continue
  fi

  if [[ -z "$TRAIN_GONE_SINCE" ]]; then
    TRAIN_GONE_SINCE=$(date +%s)
    log "ml.train not running; gone timer started"
  fi

  RUN_POST=0
  if grep -qE "${STOP_RE}" "${LOG}" 2>/dev/null; then
    RUN_POST=1
    log "stop marker in train log"
  elif (( $(date +%s) - TRAIN_GONE_SINCE >= 60 )); then
    RUN_POST=1
    log "ml.train gone >=60s (no marker required)"
  fi

  if [[ "$RUN_POST" -eq 0 ]]; then
    sleep 60
    continue
  fi

  if [[ -f "$OUT" ]]; then
    log "post_baseline_pipeline.log exists; wait_then_post likely started post — sleep 60"
    sleep 60
    continue
  fi

  if [[ -f "$LOCK" ]]; then
    log "lock $LOCK exists; skip"
    sleep 60
    continue
  fi

  log "starting post_baseline_pipeline (flock)"
  (
    flock -n 9 || exit 0
    touch "$LOCK"
    bash ml/cloud/post_baseline_pipeline.sh >> "${OUT}" 2>&1
    log "post pipeline finished rc=$?"
  ) 9>"$FLOCK"

  sleep 60
done
