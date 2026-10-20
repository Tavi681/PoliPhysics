#!/usr/bin/env bash
# Local background watcher: polls VM for 1h sanity; stops VM ONLY on SANITY_FAIL / waiter failure. DONE_TRAIN no longer stops VM (post_baseline_pipeline owns eval + VM stop).
# SANITY_ERROR (logged on VM) does NOT stop the VM or training.
INSTANCE=poliphysics-tema1-ml-train
ZONE=us-central1-b
R='/home/octav/poliphysics/Tema 1/results/ml/logs'
OUT="/Users/octav/Projects/PoliPhysics/PoliPhysics/Tema 1/results/ml/logs/sanity_watch"
mkdir -p "${OUT}"
ssh_vm() { gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="$1" 2>/dev/null; }
echo "watcher start $(date)"
SEEN=""
for i in $(seq 1 200); do   # ~16h at 5min
  sleep 300
  FAIL=$(ssh_vm "cat '${R}/sanity_fail.code' 2>/dev/null; true")
  ERR=$(ssh_vm "tail -3 '${R}/sanity_error.log' 2>/dev/null; true")
  if [[ -n "${ERR}" ]]; then
    echo "[$(date)] SANITY_ERROR (non-fatal): ${ERR}" | tee -a "${OUT}/sanity_error_notes.log"
  fi
  for CFG in gns_local_gpu gns_baseline_gpu; do
    if [[ "${SEEN}" != *"${CFG}"* ]]; then
      J=$(ssh_vm "cat '${R}/sanity_1h_${CFG}.json' 2>/dev/null; true")
      if [[ -n "${J}" ]]; then
        SEEN="${SEEN} ${CFG}"
        echo "${J}" > "${OUT}/sanity_1h_${CFG}.json"
        echo "[$(date)] sanity JSON fetched for ${CFG}"
        ssh_vm "grep -E 'SANITY_(OK|FAIL|ERROR)' '${R}/train_vm.log' | tail -5"
      fi
    fi
  done
  stop_vm() {
    local reason="$1"
    echo "[$(date)] ${reason} — stopping VM"
    ssh_vm "tail -60 '${R}/train_vm.log'" > "${OUT}/train_vm_tail_at_stop.log"
    gcloud compute instances stop "${INSTANCE}" --zone="${ZONE}"
    echo "[$(date)] VM stopped"
    exit 0
  }
  if [[ -n "${FAIL}" ]]; then
    stop_vm "SANITY_FAIL sanity_fail.code=${FAIL}"
  fi
  # Post pipeline (waiter on VM) owns eval + VM stop. Parent run_train_on_vm.sh was killed
  # on purpose, so DONE_TRAIN / "train exited" never appear; do NOT stop on DONE_TRAIN.
  W=$(ssh_vm "cat '${R}/post_waiter.log' 2>/dev/null; true")
  if echo "${W}" | grep -q 'NO stop marker'; then
    stop_vm "waiter: train ended without stop marker (crash/sanity kill) — pipeline not run"
  fi
  if echo "${W}" | grep -qE 'post pipeline rc=[1-9]'; then
    ssh_vm "tail -60 '${R}/post_baseline_pipeline.log'" > "${OUT}/post_pipeline_tail_fail.log"
    stop_vm "post pipeline FAILED"
  fi
  if echo "${W}" | grep -q 'post pipeline rc=0'; then
    echo "[$(date)] post pipeline done; pipeline stops VM itself"
    ssh_vm "tail -60 '${R}/post_baseline_pipeline.log'" > "${OUT}/post_pipeline_tail_final.log"
    exit 0
  fi
  LOG_MARK=$(ssh_vm "grep -E 'COST_CAP_SESSION_STOP|GPU util gate failed' '${R}/train_vm.log' 2>/dev/null | tail -8; true")
  if echo "${LOG_MARK}" | grep -q 'GPU util gate failed'; then
    stop_vm "GPU util gate failed"
  fi
done
