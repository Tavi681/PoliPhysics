#!/usr/bin/env bash
# Ephemeral on-demand GCP CPU for one Round-8c shard.
#   SHARD=A DETACH=true bash "Tema 2/scripts/run_round8c_export_on_gcp.sh"
# No 24 h cap. Does NOT ship .cache/mmin (item 3 runs locally after pull).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/.." && pwd)"
SHARD="${SHARD:-A}"
INSTANCE="${INSTANCE:-poliphysics-tema2-r8c-$(printf '%s' "${SHARD}" | tr '[:upper:]' '[:lower:]')}"
ZONE="${ZONE:-us-central1-a}"
MACHINE="${MACHINE:-c2-standard-16}"
PREEMPTIBLE="${PREEMPTIBLE:-false}"
DETACH="${DETACH:-true}"
JOBS="${JOBS:-16}"
EXPECTED_PROJECT="${EXPECTED_PROJECT:-authorship-verification}"
REMOTE_DIR="${REMOTE_DIR:-/home/octav/tema2}"
LOCAL_RESULTS="${LOCAL_RESULTS:-${ROOT}/paper_results/round8c/${SHARD}}"
LOCAL_LOG="${LOCAL_LOG:-${ROOT}/paper_results/round8c_${SHARD}_gcp.log}"
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
  --boot-disk-size=80GB
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

echo "Transferring Tema 2 package (no mmin cache)..."
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="mkdir -p ${REMOTE_DIR}"
TAR="${TMPDIR:-/tmp}/tema2_r8c_$$.tgz"
COPYFILE_DISABLE=1 tar --disable-copyfile -C "${ROOT}" -czf "${TAR}" \
  --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='.cache' --exclude='paper_results/*.h5' \
  --exclude='paper_results/fig_phase_maps/*.h5' \
  --exclude='paper_results/runs_round7/*.h5' \
  --exclude='paper_results/round8/*/hdf5' \
  --exclude='paper_results/round8b/local' \
  --exclude='*.egg-info' --exclude='._*' \
  netsim validation tests configs ref scripts pyproject.toml paper_results \
  COMMIT CODE_DIRTY README.md \
  2>/dev/null

gcloud compute scp "${TAR}" "${INSTANCE}:/tmp/tema2.tgz" --zone="${ZONE}"
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  mkdir -p ${REMOTE_DIR}
  tar -C ${REMOTE_DIR} -xzf /tmp/tema2.tgz 2>/dev/null
  rm -f /tmp/tema2.tgz
  chmod +x ${REMOTE_DIR}/scripts/run_round8c_on_vm.sh
"

echo "Installing deps and starting round-8c shard ${SHARD} (no wall cap)..."
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
  : > ${REMOTE_DIR}/round8c_${SHARD}.log
  export GIT_COMMIT='${GIT_COMMIT}'
  export CODE_DIRTY='${CODE_DIRTY}'
  export JOBS='${JOBS}'
  export SHARD='${SHARD}'
  nohup bash ${REMOTE_DIR}/scripts/run_round8c_on_vm.sh \
    > ${REMOTE_DIR}/round8c_${SHARD}.log 2>&1 &
  echo \$! > ${REMOTE_DIR}/job.pid
  echo NOHUP_PID=\$(cat ${REMOTE_DIR}/job.pid)
"

if [[ "${DETACH}" == "true" ]]; then
  DELETE_ON_EXIT=0
  echo "DETACH=true: shard ${SHARD} running on ${INSTANCE}"
  echo "  gcloud compute ssh ${INSTANCE} --zone=${ZONE} --command='tail -n 20 ${REMOTE_DIR}/round8c_${SHARD}.log; test -f ${REMOTE_DIR}/_ROUND8C_${SHARD}_DONE && echo DONE || echo STILL'"
  exit 0
fi

echo "Polling shard ${SHARD}..."
while true; do
  if gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --ssh-flag="-o ConnectTimeout=10" --command="
      test -f ${REMOTE_DIR}/_ROUND8C_${SHARD}_DONE
    " 2>/dev/null; then
    echo "DONE marker present."
    break
  fi
  python3 - <<PY || true
import subprocess
cmd = ["gcloud", "compute", "ssh", "${INSTANCE}", "--zone=${ZONE}",
       "--ssh-flag=-o ConnectTimeout=10",
       "--command=tail -n 6 ${REMOTE_DIR}/round8c_${SHARD}.log"]
try:
    r = subprocess.run(cmd, timeout=25, capture_output=True, text=True)
    print(r.stdout or r.stderr)
except subprocess.TimeoutExpired:
    print("ssh tail timed out (not killing)")
PY
  sleep 60
done

echo "Pulling shard ${SHARD} results..."
mkdir -p "${LOCAL_RESULTS}" "$(dirname "${LOCAL_LOG}")"
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
    tar -C ${REMOTE_DIR} -czf /tmp/tema2_r8c.tgz \
      paper_results/round8c/${SHARD} round8c_${SHARD}.log \
      2>/dev/null || true
  " || true
gcloud compute scp "${INSTANCE}:/tmp/tema2_r8c.tgz" /tmp/tema2_r8c_${SHARD}.tgz --zone="${ZONE}" || true
python3 - <<PY
from pathlib import Path
import tarfile
root = Path(r"""${ROOT}""")
p = Path("/tmp/tema2_r8c_${SHARD}.tgz")
if p.is_file():
    with tarfile.open(p) as t:
        t.extractall(root)
        print(f"extracted {p.name} members={len(t.getnames())}")
PY
if [[ -f "${ROOT}/round8c_${SHARD}.log" ]]; then
  cp "${ROOT}/round8c_${SHARD}.log" "${LOCAL_LOG}"
fi
rm -f "/tmp/tema2_r8c_${SHARD}.tgz"
echo "Pulled shard ${SHARD}; deleting VM."
