#!/usr/bin/env bash
# Ephemeral GCP CPU for Tema 1 Stage B (Algorithm alg:sweep).
# Pattern mirrors Tema 2/scripts/run_round6_export_on_gcp.sh:
#   create VM → scp code → run job → pull results → ALWAYS delete VM
#   (unless DETACH=true).
#
# Defaults to on-demand (PREEMPTIBLE=false): the Stage B grid is long-running.
# Prefer DETACH=true so closing the laptop does not tear down the VM; the job
# keeps going and you pull/delete later.
#
# Prerequisites: gcloud auth, project authorship-verification (override via env).
#
# Usage (from anywhere):
#   bash "Tema 1/scripts/run_stage_b_on_gcp.sh"
#
# Modes (MODE=...):
#   cost-probe  — 1 grid point per N × M=5; write ETA (default for first check)
#   prep        — --ksvalid + --tab-wc-m10
#   sweep       — full 11200-run production grid (resume-safe)
#   all         — prep then sweep
#
# Optional env:
#   INSTANCE ZONE MACHINE PREEMPTIBLE JOBS EXPECTED_PROJECT SERVICE_ACCOUNT
#   DETACH=true   — start job then exit WITHOUT deleting the VM
#   PULL_ML=true  — also pull results/ml_samples/ locally (default false)
#   GCS_BUCKET    — if set, print a ready gsutil command for ml_samples (e.g. gs://my-bucket/tema1)
#   DISK_GB       — boot disk size (default 100)
#   M / T_END / N_T — forwarded to stage_b.py --sweep
#   NO_HDF5=true  — pass --no-hdf5 (cheaper disk I/O; no ml_samples)
#
# After a sweep/all finishes, small artifacts are pulled automatically, then the
# script reports du -sh on ml_samples/ and LEAVES the VM up so you can decide:
# copy locally, upload to GCS, then delete the instance yourself.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # Tema 1/
REPO="$(cd "${ROOT}/.." && pwd)"                         # PoliPhysics/
INSTANCE="${INSTANCE:-poliphysics-tema1-stageb}"
ZONE="${ZONE:-us-central1-a}"
MACHINE="${MACHINE:-c2-standard-16}"
# On-demand by default: Stage B is a multi-hour / multi-day grid; spot preemption
# would force frequent resumes. Override with PREEMPTIBLE=true only if you accept that.
PREEMPTIBLE="${PREEMPTIBLE:-false}"
DETACH="${DETACH:-false}"
MODE="${MODE:-cost-probe}"
JOBS="${JOBS:-16}"
DISK_GB="${DISK_GB:-100}"
PULL_ML="${PULL_ML:-false}"
NO_HDF5="${NO_HDF5:-false}"
GCS_BUCKET="${GCS_BUCKET:-}"                 # e.g. gs://my-bucket/tema1-stageb
M="${M:-200}"
T_END="${T_END:-60}"
N_T="${N_T:-100}"
EXPECTED_PROJECT="${EXPECTED_PROJECT:-authorship-verification}"
REMOTE_DIR="${REMOTE_DIR:-/home/octav/tema1}"
LOCAL_RESULTS="${LOCAL_RESULTS:-${ROOT}/results}"
LOCAL_LOG="${LOCAL_LOG:-${ROOT}/results/stage_b_gcp.log}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-ephemeral-benchmark@authorship-verification.iam.gserviceaccount.com}"

export CLOUDSDK_CORE_DISABLE_FILE_LOGGING=1
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
echo "gcloud account: ${ACCOUNT:-<none>} | project: ${PROJECT:-<none>} | zone: ${ZONE} | instance: ${INSTANCE}"
echo "machine: ${MACHINE} preemptible=${PREEMPTIBLE} detach=${DETACH} jobs=${JOBS} mode=${MODE}"
echo "disk=${DISK_GB}GB pull_ml=${PULL_ML} no_hdf5=${NO_HDF5} M=${M} t_end=${T_END}"

if [[ -z "${PROJECT}" || "${PROJECT}" != "${EXPECTED_PROJECT}" ]]; then
  echo "ERROR: need project ${EXPECTED_PROJECT}, got '${PROJECT:-<none>}'" >&2
  exit 1
fi

case "${MODE}" in
  cost-probe|prep|sweep|all) ;;
  *)
    echo "ERROR: MODE must be cost-probe|prep|sweep|all, got '${MODE}'" >&2
    exit 1
    ;;
esac

if gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" &>/dev/null; then
  echo "ERROR: instance ${INSTANCE} already exists — refuse to clobber." >&2
  echo "Delete it, or set INSTANCE=... to a new name. For resume on a live VM, ssh and re-run stage_b.py --sweep." >&2
  exit 1
fi

TAR=""
# With DETACH=true, never auto-delete (even if setup fails) so we can inspect logs.
DELETE_ON_EXIT=1
if [[ "${DETACH}" == "true" ]]; then
  DELETE_ON_EXIT=0
fi
cleanup() {
  local ec=$?
  rm -f "${TAR}"
  if [[ "${DELETE_ON_EXIT}" == "1" ]]; then
    echo "Deleting instance ${INSTANCE} in ${ZONE} (mandatory cleanup)..."
    gcloud compute instances delete "${INSTANCE}" --zone="${ZONE}" --quiet || true
  else
    echo "Leaving instance ${INSTANCE} running (DETACH/no-delete)."
  fi
  return "${ec}"
}
trap cleanup EXIT

# Ubuntu 24.04 ships Python >= 3.12 (ballooning requires >=3.11; 22.04 is 3.10).
CREATE_ARGS=(
  --zone="${ZONE}"
  --machine-type="${MACHINE}"
  --image-family=ubuntu-2404-lts-amd64
  --image-project=ubuntu-os-cloud
  --boot-disk-size="${DISK_GB}GB"
  --boot-disk-type=pd-balanced
  --service-account="${SERVICE_ACCOUNT}"
  --scopes=https://www.googleapis.com/auth/cloud-platform
)
if [[ "${PREEMPTIBLE}" == "true" ]]; then
  CREATE_ARGS+=(--provisioning-model=SPOT --instance-termination-action=DELETE)
fi

echo "Creating ${INSTANCE}..."
gcloud compute instances create "${INSTANCE}" "${CREATE_ARGS[@]}"

echo "Waiting for RUNNING + SSH..."
for _ in $(seq 1 60); do
  STATUS="$(gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" --format='get(status)' 2>/dev/null || true)"
  [[ "${STATUS}" == "RUNNING" ]] && break
  sleep 5
done
for _ in $(seq 1 36); do
  gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="true" 2>/dev/null && break
  sleep 5
done

echo "Transferring Tema 1 package (no .venv / ml_samples / equal_t_cache)..."
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="mkdir -p ${REMOTE_DIR}"

git -C "${REPO}" rev-parse HEAD > "${ROOT}/COMMIT" 2>/dev/null || echo "unknown" > "${ROOT}/COMMIT"
echo "Packed commit: $(cat "${ROOT}/COMMIT")"

TAR="${TMPDIR:-/tmp}/tema1_stageb_$$.tgz"
# Include existing sweep.csv / tab_wc / meta so a re-packed launch can resume.
tar -C "${ROOT}" -czf "${TAR}" \
  --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='.cache' --exclude='*.egg-info' \
  --exclude='results/ml_samples' --exclude='results/ml_samples_smoke' \
  --exclude='results/equal_t_cache' --exclude='results/equal_t_cache_smoke' \
  --exclude='results/figs' --exclude='results/pre_stageA' \
  --exclude='paper' \
  ballooning scripts tests pyproject.toml README.md COMMIT \
  results

gcloud compute scp "${TAR}" "${INSTANCE}:/tmp/tema1.tgz" --zone="${ZONE}"
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  mkdir -p ${REMOTE_DIR}
  tar -C ${REMOTE_DIR} -xzf /tmp/tema1.tgz
  rm -f /tmp/tema1.tgz
  mkdir -p ${REMOTE_DIR}/results/ml_samples ${REMOTE_DIR}/results/snapshots
"

# Build remote command for the selected MODE.
HDF5_FLAG=""
if [[ "${NO_HDF5}" == "true" ]]; then
  HDF5_FLAG="--no-hdf5"
fi

REMOTE_JOB=""
DONE_FILE=""
case "${MODE}" in
  cost-probe)
    REMOTE_JOB="python -u scripts/stage_b.py --cost-probe --workers ${JOBS}"
    DONE_FILE="results/cost_probe_DONE"
    ;;
  prep)
    REMOTE_JOB="python -u scripts/stage_b.py --ksvalid --tab-wc-m10 --workers ${JOBS}"
    DONE_FILE="results/prep_DONE"
    ;;
  sweep)
    REMOTE_JOB="python -u scripts/stage_b.py --sweep --workers ${JOBS} --M ${M} --t-end ${T_END} --N-t ${N_T} ${HDF5_FLAG}"
    DONE_FILE="results/sweep_DONE"
    ;;
  all)
    REMOTE_JOB="python -u scripts/stage_b.py --ksvalid --tab-wc-m10 --workers ${JOBS} && python -u scripts/stage_b.py --sweep --workers ${JOBS} --M ${M} --t-end ${T_END} --N-t ${N_T} ${HDF5_FLAG}"
    DONE_FILE="results/sweep_DONE"
    ;;
esac

echo "Installing deps and starting MODE=${MODE}..."
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  export DEBIAN_FRONTEND=noninteractive
  export PYTHONUNBUFFERED=1
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv python3-pip build-essential
  cd ${REMOTE_DIR}
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -U pip wheel
  pip install -q -e .
  mkdir -p results
  : > ${REMOTE_DIR}/stage_b.log
  # Clear stale DONE markers from a previous pack (except we keep resume CSV).
  rm -f results/sweep_DONE results/cost_probe_DONE results/prep_DONE
  nohup bash -lc '
    set -euo pipefail
    cd ${REMOTE_DIR}
    source .venv/bin/activate
    ${REMOTE_JOB}
    # prep mode has no native DONE file
    if [[ \"${MODE}\" == \"prep\" ]]; then
      echo ok > results/prep_DONE
    fi
  ' > ${REMOTE_DIR}/stage_b.log 2>&1 &
  echo \$! > ${REMOTE_DIR}/job.pid
  echo NOHUP_PID=\$(cat ${REMOTE_DIR}/job.pid)
  echo REMOTE_JOB=${REMOTE_JOB}
"

_print_ml_decision_help() {
  echo ""
  echo "=== ml_samples decision (VM still up) ==="
  echo "Check size on VM:"
  echo "  gcloud compute ssh ${INSTANCE} --zone=${ZONE} --command='du -sh ${REMOTE_DIR}/results/ml_samples 2>/dev/null; find ${REMOTE_DIR}/results/ml_samples -name \"*.h5\" 2>/dev/null | wc -l'"
  echo "Option A — pull locally:"
  echo "  mkdir -p \"${LOCAL_RESULTS}/ml_samples\""
  echo "  gcloud compute scp --recurse ${INSTANCE}:${REMOTE_DIR}/results/ml_samples/. \"${LOCAL_RESULTS}/ml_samples/\" --zone=${ZONE}"
  if [[ -n "${GCS_BUCKET}" ]]; then
    echo "Option B — upload to GCS (${GCS_BUCKET}):"
    echo "  gcloud compute ssh ${INSTANCE} --zone=${ZONE} --command='gsutil -m cp -r ${REMOTE_DIR}/results/ml_samples ${GCS_BUCKET}/'"
  else
    echo "Option B — upload to GCS (set GCS_BUCKET=gs://.../tema1-stageb):"
    echo "  gcloud compute ssh ${INSTANCE} --zone=${ZONE} --command='gsutil -m cp -r ${REMOTE_DIR}/results/ml_samples gs://YOUR_BUCKET/tema1-stageb/'"
  fi
  echo "Then delete the VM:"
  echo "  gcloud compute instances delete ${INSTANCE} --zone=${ZONE} --quiet"
}

if [[ "${DETACH}" == "true" ]]; then
  echo "DETACH=true: job running on ${INSTANCE}; local script exits."
  echo "Status:"
  echo "  gcloud compute ssh ${INSTANCE} --zone=${ZONE} --command='tail -n 30 ${REMOTE_DIR}/stage_b.log; ps -p \$(cat ${REMOTE_DIR}/job.pid) || echo DONE'"
  echo "  gcloud compute ssh ${INSTANCE} --zone=${ZONE} --command='tail -n 20 ${REMOTE_DIR}/results/sweep_log.txt'"
  echo "When finished, pull small artifacts first:"
  echo "  gcloud compute scp ${INSTANCE}:${REMOTE_DIR}/results/sweep.csv \"${LOCAL_RESULTS}/\" --zone=${ZONE}"
  echo "  gcloud compute scp ${INSTANCE}:${REMOTE_DIR}/results/tab_phase.csv \"${LOCAL_RESULTS}/\" --zone=${ZONE}"
  echo "  gcloud compute scp ${INSTANCE}:${REMOTE_DIR}/results/sweep_log.txt \"${LOCAL_RESULTS}/\" --zone=${ZONE}"
  echo "  gcloud compute scp ${INSTANCE}:${REMOTE_DIR}/results/meta.json \"${LOCAL_RESULTS}/\" --zone=${ZONE}"
  echo "  gcloud compute scp --recurse ${INSTANCE}:${REMOTE_DIR}/results/snapshots/. \"${LOCAL_RESULTS}/snapshots/\" --zone=${ZONE}"
  echo "  gcloud compute scp ${INSTANCE}:${REMOTE_DIR}/stage_b.log \"${LOCAL_LOG}\" --zone=${ZONE}"
  _print_ml_decision_help
  exit 0
fi

echo "Polling for ${DONE_FILE}..."
while true; do
  if gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
      test -f ${REMOTE_DIR}/${DONE_FILE} \\
      && ! ps -p \$(cat ${REMOTE_DIR}/job.pid) >/dev/null 2>&1
    " 2>/dev/null; then
    echo "[watch] ${DONE_FILE} present and job exited"
    break
  fi
  gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" \
    --command="tail -n 15 ${REMOTE_DIR}/stage_b.log; echo '---'; tail -n 8 ${REMOTE_DIR}/results/sweep_log.txt 2>/dev/null || true" \
    2>/dev/null || true
  if ! gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" \
      --command="ps -p \$(cat ${REMOTE_DIR}/job.pid) -o pid= >/dev/null" 2>/dev/null; then
    if gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" \
        --command="test -f ${REMOTE_DIR}/${DONE_FILE}" 2>/dev/null; then
      echo "[watch] process exited with ${DONE_FILE}"
      break
    fi
    echo "ERROR: job process died early" >&2
    gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" \
      --command="tail -n 100 ${REMOTE_DIR}/stage_b.log" || true
    exit 1
  fi
  sleep 60
done

mkdir -p "${LOCAL_RESULTS}/snapshots"
echo "Pulling small/critical results → ${LOCAL_RESULTS}..."
for f in sweep.csv tab_phase.csv sweep_log.txt meta.json tab_wc.csv \
         fig_ksvalid.csv fig_ksvalid_pdf.csv sweep_DONE cost_probe_DONE prep_DONE; do
  gcloud compute scp \
    "${INSTANCE}:${REMOTE_DIR}/results/${f}" \
    "${LOCAL_RESULTS}/" \
    --zone="${ZONE}" 2>/dev/null || true
done
gcloud compute scp --recurse \
  "${INSTANCE}:${REMOTE_DIR}/results/snapshots/." \
  "${LOCAL_RESULTS}/snapshots/" \
  --zone="${ZONE}" 2>/dev/null || true
gcloud compute scp \
  "${INSTANCE}:${REMOTE_DIR}/stage_b.log" \
  "${LOCAL_LOG}" \
  --zone="${ZONE}" || true

# Report ml_samples size before any delete decision.
ML_DU="$(gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" \
  --command="du -sh ${REMOTE_DIR}/results/ml_samples 2>/dev/null || echo '0	(no ml_samples)'" \
  2>/dev/null || echo "unknown")"
ML_N="$(gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" \
  --command="find ${REMOTE_DIR}/results/ml_samples -name '*.h5' 2>/dev/null | wc -l" \
  2>/dev/null || echo "0")"
echo "ml_samples on VM: ${ML_DU}  (n_h5=${ML_N})"

# Sweep/all may have a large ml_samples tree — leave the VM up so we can choose
# local pull vs GCS before paying for another transfer or deleting data.
if [[ "${MODE}" == "sweep" || "${MODE}" == "all" ]]; then
  if [[ "${PULL_ML}" == "true" ]]; then
    echo "PULL_ML=true: pulling ml_samples locally..."
    mkdir -p "${LOCAL_RESULTS}/ml_samples"
    gcloud compute scp --recurse \
      "${INSTANCE}:${REMOTE_DIR}/results/ml_samples/." \
      "${LOCAL_RESULTS}/ml_samples/" \
      --zone="${ZONE}" || true
    echo "Done. Small results + ml_samples in ${LOCAL_RESULTS}."
    echo "Instance will be deleted via EXIT trap."
  else
    DELETE_ON_EXIT=0
    echo "Done. Small results in ${LOCAL_RESULTS}."
    echo "Leaving VM up so you can decide what to do with ml_samples."
    _print_ml_decision_help
  fi
else
  # cost-probe / prep: no large ml_samples expected
  echo "Done. Results in ${LOCAL_RESULTS}. Instance will be deleted via EXIT trap."
fi
