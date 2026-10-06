#!/usr/bin/env bash
# Ephemeral on-demand GCP CPU for one Round-8 shard.
#   SHARD=pilot DETACH=true bash "Tema 2/scripts/run_round8_export_on_gcp.sh"
# No 24 h cap. Does not ship .cache/mmin.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/.." && pwd)"
SHARD="${SHARD:-pilot}"
INSTANCE="${INSTANCE:-poliphysics-tema2-r8-${SHARD}}"
ZONE="${ZONE:-us-central1-a}"
MACHINE="${MACHINE:-c2-standard-16}"
PREEMPTIBLE="${PREEMPTIBLE:-false}"
DETACH="${DETACH:-true}"
JOBS="${JOBS:-16}"
EXPECTED_PROJECT="${EXPECTED_PROJECT:-authorship-verification}"
REMOTE_DIR="${REMOTE_DIR:-/home/octav/tema2}"
LOCAL_RESULTS="${LOCAL_RESULTS:-${ROOT}/paper_results/round8/${SHARD}}"
LOCAL_LOG="${LOCAL_LOG:-${ROOT}/paper_results/round8_${SHARD}_gcp.log}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-ephemeral-benchmark@authorship-verification.iam.gserviceaccount.com}"

export CLOUDSDK_CORE_DISABLE_FILE_LOGGING=1
export CLOUDSDK_CORE_DISABLE_PROMPTS=1
export COPYFILE_DISABLE=1

PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
echo "gcloud account: ${ACCOUNT:-<none>} | project: ${PROJECT:-<none>} | zone: ${ZONE} | instance: ${INSTANCE}"
echo "shard=${SHARD} machine: ${MACHINE} detach=${DETACH} jobs=${JOBS}"

if [[ -z "${PROJECT}" || "${PROJECT}" != "${EXPECTED_PROJECT}" ]]; then
  echo "ERROR: need project ${EXPECTED_PROJECT}, got '${PROJECT:-<none>}'" >&2
  exit 1
fi

if gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" &>/dev/null; then
  echo "ERROR: instance ${INSTANCE} already exists — refuse to clobber." >&2
  exit 1
fi

cd "${REPO}"
CODE_DIRTY=false
if ! git diff --quiet -- "Tema 2/netsim" "Tema 2/validation" "Tema 2/tests" "Tema 2/configs"; then
  CODE_DIRTY=true
fi
if [[ -n "$(git ls-files --others --exclude-standard -- "Tema 2/netsim" "Tema 2/validation" "Tema 2/tests" "Tema 2/configs")" ]]; then
  CODE_DIRTY=true
fi
GIT_COMMIT="$(git rev-parse HEAD)"
echo "GIT_COMMIT=${GIT_COMMIT} CODE_DIRTY=${CODE_DIRTY}"
printf '%s\n' "${GIT_COMMIT}" > "${ROOT}/COMMIT"
printf '%s\n' "${CODE_DIRTY}" > "${ROOT}/CODE_DIRTY"

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

echo "Transferring Tema 2 package WITHOUT mmin cache..."
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="mkdir -p ${REMOTE_DIR}"
TAR="${TMPDIR:-/tmp}/tema2_r8_$$.tgz"
tar -C "${ROOT}" -czf "${TAR}" \
  --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='.cache' --exclude='paper_results/*.h5' \
  --exclude='paper_results/fig_phase_maps/*.h5' \
  --exclude='paper_results/runs_round7/*.h5' \
  --exclude='*.egg-info' --exclude='._*' \
  netsim validation tests configs ref scripts pyproject.toml paper_results \
  COMMIT CODE_DIRTY README.md

gcloud compute scp "${TAR}" "${INSTANCE}:/tmp/tema2.tgz" --zone="${ZONE}"
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  mkdir -p ${REMOTE_DIR}
  tar -C ${REMOTE_DIR} -xzf /tmp/tema2.tgz
  rm -f /tmp/tema2.tgz
  chmod +x ${REMOTE_DIR}/scripts/run_round8_on_vm.sh
"

echo "Installing deps and starting round-8 shard ${SHARD} (no wall cap)..."
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
  : > ${REMOTE_DIR}/round8_${SHARD}.log
  export GIT_COMMIT='${GIT_COMMIT}'
  export CODE_DIRTY='${CODE_DIRTY}'
  export JOBS='${JOBS}'
  export SHARD='${SHARD}'
  nohup bash ${REMOTE_DIR}/scripts/run_round8_on_vm.sh \
    > ${REMOTE_DIR}/round8_${SHARD}.log 2>&1 &
  echo \$! > ${REMOTE_DIR}/job.pid
  echo NOHUP_PID=\$(cat ${REMOTE_DIR}/job.pid)
"

if [[ "${DETACH}" == "true" ]]; then
  DELETE_ON_EXIT=0
  echo "DETACH=true: shard ${SHARD} running on ${INSTANCE}"
  echo "  gcloud compute ssh ${INSTANCE} --zone=${ZONE} --command='tail -n 20 ${REMOTE_DIR}/round8_${SHARD}.log; test -f ${REMOTE_DIR}/_ROUND8_${SHARD}_DONE && echo DONE || (ps -p \$(cat ${REMOTE_DIR}/job.pid) || echo DEAD)'"
  exit 0
fi

echo "Polling shard ${SHARD}..."
STALE_WARNED=0
LAST_BYTES=0
LAST_CHANGE=$(date +%s)
while true; do
  if gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
      test -f ${REMOTE_DIR}/_ROUND8_${SHARD}_DONE
    " 2>/dev/null; then
    echo "DONE marker present."
    break
  fi
  gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" \
    --command="tail -n 6 ${REMOTE_DIR}/round8_${SHARD}.log" 2>/dev/null || true
  BYTES=$(gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" \
    --command="stat -c%s ${REMOTE_DIR}/round8_${SHARD}.log 2>/dev/null || echo 0" \
    2>/dev/null || echo 0)
  NOW=$(date +%s)
  if [[ "${BYTES}" != "${LAST_BYTES}" ]]; then
    LAST_BYTES="${BYTES}"
    LAST_CHANGE="${NOW}"
    STALE_WARNED=0
  elif (( NOW - LAST_CHANGE > 7200 && STALE_WARNED == 0 )); then
    echo "WARNING: shard ${SHARD} log unchanged for 2 h — not killing." | tee -a "${LOCAL_LOG}"
    STALE_WARNED=1
  fi
  sleep 60
done

echo "Pulling shard ${SHARD} results..."
mkdir -p "${LOCAL_RESULTS}" "$(dirname "${LOCAL_LOG}")"
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
    tar -C ${REMOTE_DIR} -czf /tmp/tema2_r8.tgz paper_results/round8/${SHARD} round8_${SHARD}.log \
      paper_results/params_used.csv paper_results/tab_conv.csv paper_results/fig_etaa.csv 2>/dev/null || \
    tar -C ${REMOTE_DIR} -czf /tmp/tema2_r8.tgz paper_results/round8/${SHARD} round8_${SHARD}.log
  " || true
gcloud compute scp "${INSTANCE}:/tmp/tema2_r8.tgz" /tmp/tema2_r8_${SHARD}.tgz --zone="${ZONE}" || true
python3 - <<PY
from pathlib import Path
import tarfile
root = Path(r"""${ROOT}""")
p = Path("/tmp/tema2_r8_${SHARD}.tgz")
if p.is_file():
    with tarfile.open(p) as t:
        t.extractall(root)
        print(f"extracted {p.name} members={len(t.getnames())}")
PY
if [[ -f "${ROOT}/round8_${SHARD}.log" ]]; then
  cp "${ROOT}/round8_${SHARD}.log" "${LOCAL_LOG}"
fi
rm -f "/tmp/tema2_r8_${SHARD}.tgz"
echo "Pulled shard ${SHARD}; deleting VM."
# trap cleanup deletes because DELETE_ON_EXIT=1
