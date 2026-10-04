#!/usr/bin/env bash
# Ephemeral on-demand GCP CPU for Tema 2 round-7 export.
#   create VM → scp code → run export (24h wall) → pull results+cache → delete VM.
#
# Usage:
#   DETACH=true bash "Tema 2/scripts/run_round7_export_on_gcp.sh"
#
# Optional env: INSTANCE ZONE MACHINE PREEMPTIBLE JOBS DETACH
#   PREEMPTIBLE=false (default)  — on-demand; spot preempted r6.
#   DETACH=true                  — start job then exit WITHOUT deleting the VM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # Tema 2/
REPO="$(cd "${ROOT}/.." && pwd)"                         # PoliPhysics/
INSTANCE="${INSTANCE:-poliphysics-tema2-r7}"
ZONE="${ZONE:-us-central1-a}"
MACHINE="${MACHINE:-c2-standard-16}"
PREEMPTIBLE="${PREEMPTIBLE:-false}"
DETACH="${DETACH:-true}"
JOBS="${JOBS:-16}"
WALL_S="${WALL_S:-86400}"
EXPECTED_PROJECT="${EXPECTED_PROJECT:-authorship-verification}"
REMOTE_DIR="${REMOTE_DIR:-/home/octav/tema2}"
LOCAL_RESULTS="${LOCAL_RESULTS:-${ROOT}/paper_results}"
LOCAL_CACHE="${LOCAL_CACHE:-${ROOT}/.cache}"
LOCAL_LOG="${LOCAL_LOG:-${ROOT}/paper_results/round7_export_gcp.log}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-ephemeral-benchmark@authorship-verification.iam.gserviceaccount.com}"

export CLOUDSDK_CORE_DISABLE_FILE_LOGGING=1
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
echo "gcloud account: ${ACCOUNT:-<none>} | project: ${PROJECT:-<none>} | zone: ${ZONE} | instance: ${INSTANCE}"
echo "machine: ${MACHINE} preemptible=${PREEMPTIBLE} detach=${DETACH} jobs=${JOBS} wall=${WALL_S}s"

if [[ -z "${PROJECT}" || "${PROJECT}" != "${EXPECTED_PROJECT}" ]]; then
  echo "ERROR: need project ${EXPECTED_PROJECT}, got '${PROJECT:-<none>}'" >&2
  exit 1
fi

if gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" &>/dev/null; then
  echo "ERROR: instance ${INSTANCE} already exists — refuse to clobber." >&2
  exit 1
fi

TAR=""
DELETE_ON_EXIT=1
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

# Provenance from the *local* tree (VM has no .git).
GIT_COMMIT="$(git -C "${REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
if git -C "${REPO}" diff --quiet HEAD -- \
      "Tema 2/netsim" "Tema 2/validation" "Tema 2/tests" "Tema 2/configs" \
    && [[ -z "$(git -C "${REPO}" ls-files --others --exclude-standard -- \
      "Tema 2/netsim" "Tema 2/validation" "Tema 2/tests" "Tema 2/configs")" ]]; then
  CODE_DIRTY=false
else
  CODE_DIRTY=true
fi
printf '%s\n' "${GIT_COMMIT}" > "${ROOT}/COMMIT"
printf '%s\n' "${CODE_DIRTY}" > "${ROOT}/CODE_DIRTY"
echo "Packed commit=${GIT_COMMIT} code_dirty=${CODE_DIRTY}"

echo "Transferring Tema 2 package (no .venv / existing mmin cache)..."
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="mkdir -p ${REMOTE_DIR}"

TAR="${TMPDIR:-/tmp}/tema2_r7_$$.tgz"
tar -C "${ROOT}" -czf "${TAR}" \
  --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='.cache' --exclude='paper_results/*.h5' \
  --exclude='paper_results/fig_phase_maps/*.h5' \
  --exclude='paper_results/runs_round7/*.h5' \
  --exclude='*.egg-info' \
  netsim validation tests configs ref scripts pyproject.toml paper_results COMMIT CODE_DIRTY

gcloud compute scp "${TAR}" "${INSTANCE}:/tmp/tema2.tgz" --zone="${ZONE}"
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  mkdir -p ${REMOTE_DIR}
  tar -C ${REMOTE_DIR} -xzf /tmp/tema2.tgz
  rm -f /tmp/tema2.tgz
  chmod +x ${REMOTE_DIR}/scripts/run_round7_on_vm.sh
"

echo "Installing deps and starting round-7 job (wall ${WALL_S}s)..."
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
  : > ${REMOTE_DIR}/round7_export.log
  export GIT_COMMIT='${GIT_COMMIT}'
  export CODE_DIRTY='${CODE_DIRTY}'
  export JOBS='${JOBS}'
  nohup timeout ${WALL_S} bash ${REMOTE_DIR}/scripts/run_round7_on_vm.sh \
    > ${REMOTE_DIR}/round7_export.log 2>&1 &
  echo \$! > ${REMOTE_DIR}/job.pid
  echo NOHUP_PID=\$(cat ${REMOTE_DIR}/job.pid)
"

pull_results() {
  mkdir -p "${LOCAL_RESULTS}" "${LOCAL_CACHE}"
  echo "Pulling paper_results → ${LOCAL_RESULTS}..."
  gcloud compute scp --recurse \
    "${INSTANCE}:${REMOTE_DIR}/paper_results/." \
    "${LOCAL_RESULTS}/" \
    --zone="${ZONE}" || true
  echo "Pulling .cache → ${LOCAL_CACHE}..."
  gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
      test -d ${REMOTE_DIR}/.cache && tar -C ${REMOTE_DIR} -czf /tmp/tema2_cache.tgz .cache
    " 2>/dev/null || true
  gcloud compute scp \
    "${INSTANCE}:/tmp/tema2_cache.tgz" \
    "/tmp/tema2_r7_cache.tgz" \
    --zone="${ZONE}" 2>/dev/null && \
    tar -C "${ROOT}" -xzf /tmp/tema2_r7_cache.tgz || true
  rm -f /tmp/tema2_r7_cache.tgz
  gcloud compute scp \
    "${INSTANCE}:${REMOTE_DIR}/round7_export.log" \
    "${LOCAL_LOG}" \
    --zone="${ZONE}" || true
}

if [[ "${DETACH}" == "true" ]]; then
  DELETE_ON_EXIT=0
  echo "DETACH=true: export running on ${INSTANCE}; local script exits."
  echo "Status:"
  echo "  gcloud compute ssh ${INSTANCE} --zone=${ZONE} --command='tail -n 20 ${REMOTE_DIR}/round7_export.log; test -f ${REMOTE_DIR}/_ROUND7_DONE && echo DONE || (ps -p \$(cat ${REMOTE_DIR}/job.pid) || echo DEAD)'"
  echo "Pull + delete when finished:"
  echo "  gcloud compute scp --recurse ${INSTANCE}:${REMOTE_DIR}/paper_results/. \"${LOCAL_RESULTS}/\" --zone=${ZONE}"
  echo "  gcloud compute instances delete ${INSTANCE} --zone=${ZONE} --quiet"
  exit 0
fi

echo "Polling for export completion..."
while true; do
  if gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
      test -f ${REMOTE_DIR}/_ROUND7_DONE
    " 2>/dev/null; then
    echo "[watch] _ROUND7_DONE"
    break
  fi
  gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" \
    --command="tail -n 12 ${REMOTE_DIR}/round7_export.log" 2>/dev/null || true
  if ! gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" \
      --command="ps -p \$(cat ${REMOTE_DIR}/job.pid) -o pid= >/dev/null" 2>/dev/null; then
    if gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" \
        --command="test -f ${REMOTE_DIR}/_ROUND7_DONE || grep -q 'round7 on-vm done' ${REMOTE_DIR}/round7_export.log" 2>/dev/null; then
      echo "[watch] process exited after completion marker"
      break
    fi
    echo "ERROR: export process died early" >&2
    gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" \
      --command="tail -n 80 ${REMOTE_DIR}/round7_export.log" || true
    pull_results
    exit 1
  fi
  sleep 60
done

pull_results
echo "Done. Results in ${LOCAL_RESULTS}. Instance will be deleted via EXIT trap."
echo "Remaining instances:"
gcloud compute instances list || true
