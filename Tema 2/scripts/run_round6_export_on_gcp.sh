#!/usr/bin/env bash
# Ephemeral GCP CPU (preemptible/spot) for Tema 2 round-6 selective export.
# Pattern mirrors hunter-distance/scripts/run_*_on_gcp.sh:
#   create VM → scp code → run export → pull paper_results → ALWAYS delete VM.
#
# Exports only: tab_mmin.csv, fig_phase_maps/, fig_etaa.csv, fig_daf_ramp.csv,
# and dyn_runs.csv (the rows the maps use). Uses mmin HDF5 cache on the VM.
#
# Prerequisites: gcloud auth, project authorship-verification (override via env).
# Usage (from anywhere):
#   bash "Tema 2/scripts/run_round6_export_on_gcp.sh"
#
# Optional env:
#   INSTANCE ZONE MACHINE PREEMPTIBLE JOBS EXPECTED_PROJECT SERVICE_ACCOUNT
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # Tema 2/
REPO="$(cd "${ROOT}/.." && pwd)"                         # PoliPhysics/
INSTANCE="${INSTANCE:-poliphysics-tema2-r6}"
ZONE="${ZONE:-us-central1-a}"
MACHINE="${MACHINE:-c2-standard-16}"
PREEMPTIBLE="${PREEMPTIBLE:-true}"          # spot-like; set false for on-demand
JOBS="${JOBS:-16}"
EXPECTED_PROJECT="${EXPECTED_PROJECT:-authorship-verification}"
REMOTE_DIR="${REMOTE_DIR:-/home/octav/tema2}"
LOCAL_RESULTS="${LOCAL_RESULTS:-${ROOT}/paper_results}"
LOCAL_LOG="${LOCAL_LOG:-${ROOT}/paper_results/round6_export_gcp.log}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-ephemeral-benchmark@authorship-verification.iam.gserviceaccount.com}"

export CLOUDSDK_CORE_DISABLE_FILE_LOGGING=1
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
echo "gcloud account: ${ACCOUNT:-<none>} | project: ${PROJECT:-<none>} | zone: ${ZONE} | instance: ${INSTANCE}"
echo "machine: ${MACHINE} preemptible=${PREEMPTIBLE} jobs=${JOBS}"

if [[ -z "${PROJECT}" || "${PROJECT}" != "${EXPECTED_PROJECT}" ]]; then
  echo "ERROR: need project ${EXPECTED_PROJECT}, got '${PROJECT:-<none>}'" >&2
  exit 1
fi

if gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" &>/dev/null; then
  echo "ERROR: instance ${INSTANCE} already exists — refuse to clobber." >&2
  exit 1
fi

TAR=""
cleanup() {
  local ec=$?
  rm -f "${TAR}"
  echo "Deleting instance ${INSTANCE} in ${ZONE} (mandatory cleanup)..."
  gcloud compute instances delete "${INSTANCE}" --zone="${ZONE}" --quiet || true
  return "${ec}"
}
trap cleanup EXIT

CREATE_ARGS=(
  --zone="${ZONE}"
  --machine-type="${MACHINE}"
  --image-family=ubuntu-2204-lts
  --image-project=ubuntu-os-cloud
  --boot-disk-size=50GB
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

echo "Transferring Tema 2 package (no .venv / mmin cache)..."
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="mkdir -p ${REMOTE_DIR}"

TAR="${TMPDIR:-/tmp}/tema2_r6_$$.tgz"
tar -C "${ROOT}" -czf "${TAR}" \
  --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='.cache' --exclude='paper_results/*.h5' \
  --exclude='paper_results/fig_phase_maps/*.h5' \
  --exclude='*.egg-info' \
  netsim validation tests configs ref pyproject.toml paper_results

gcloud compute scp "${TAR}" "${INSTANCE}:/tmp/tema2.tgz" --zone="${ZONE}"
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  mkdir -p ${REMOTE_DIR}
  tar -C ${REMOTE_DIR} -xzf /tmp/tema2.tgz
  rm -f /tmp/tema2.tgz
"

echo "Installing deps and starting selective export..."
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
  pip install -q -e '.[test]'
  # Round-6 selective re-export (not a full SPEC dump).
  : > ${REMOTE_DIR}/round6_export.log
  nohup python -u -m validation.export_paper --full --jobs ${JOBS} --allow-dirty \
    --only tab_mmin.csv,fig_etaa.csv,fig_daf_ramp.csv,dyn_runs.csv,fig_phase_maps/ \
    > ${REMOTE_DIR}/round6_export.log 2>&1 &
  echo \$! > ${REMOTE_DIR}/job.pid
  echo NOHUP_PID=\$(cat ${REMOTE_DIR}/job.pid)
"

echo "Polling for ${REMOTE_DIR}/paper_results/_ROUND6_DONE (touched at end)..."
# export_paper has no _DONE file — watch the log for completion + dead process.
while true; do
  if gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
      grep -q '\\[ok\\].*fig_phase_maps' ${REMOTE_DIR}/round6_export.log 2>/dev/null \\
      && ! ps -p \$(cat ${REMOTE_DIR}/job.pid) >/dev/null 2>&1
    " 2>/dev/null; then
    echo "[watch] export finished"
    break
  fi
  gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" \
    --command="tail -n 12 ${REMOTE_DIR}/round6_export.log" 2>/dev/null || true
  if ! gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" \
      --command="ps -p \$(cat ${REMOTE_DIR}/job.pid) -o pid= >/dev/null" 2>/dev/null; then
    # Process died — accept if last exporters ok, else fail.
    if gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" \
        --command="grep -q '\\[ok\\].*tab_mmin' ${REMOTE_DIR}/round6_export.log" 2>/dev/null; then
      echo "[watch] process exited after tab_mmin ok"
      break
    fi
    echo "ERROR: export process died early" >&2
    gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" \
      --command="tail -n 80 ${REMOTE_DIR}/round6_export.log" || true
    exit 1
  fi
  sleep 60
done

mkdir -p "${LOCAL_RESULTS}"
echo "Pulling paper_results → ${LOCAL_RESULTS}..."
gcloud compute scp --recurse \
  "${INSTANCE}:${REMOTE_DIR}/paper_results/." \
  "${LOCAL_RESULTS}/" \
  --zone="${ZONE}" || true
gcloud compute scp \
  "${INSTANCE}:${REMOTE_DIR}/round6_export.log" \
  "${LOCAL_LOG}" \
  --zone="${ZONE}" || true

echo "Done. Results in ${LOCAL_RESULTS}. Instance will be deleted via EXIT trap."
