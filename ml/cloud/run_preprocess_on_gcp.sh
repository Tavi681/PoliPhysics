#!/usr/bin/env bash
# Create c2-standard-16 (us-central1-a, 100GB PD), stream-preprocess all 8200 runs, upload, verify.
#   DETACH=true bash ml/cloud/run_preprocess_on_gcp.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTANCE="${INSTANCE:-poliphysics-tema1-ml-preprocess}"
ZONE="${ZONE:-us-central1-a}"
MACHINE="${MACHINE:-c2-standard-16}"
DISK_GB="${DISK_GB:-100}"
DETACH="${DETACH:-true}"
WORKERS="${WORKERS:-16}"
EXPECTED_PROJECT="${EXPECTED_PROJECT:-authorship-verification}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-ephemeral-benchmark@authorship-verification.iam.gserviceaccount.com}"
REMOTE_DIR="${REMOTE_DIR:-/home/octav/poliphysics}"
LOCAL_LOG="${LOCAL_LOG:-${REPO}/Tema 1/results/ml/preprocess_gcp.log}"

export CLOUDSDK_CORE_DISABLE_FILE_LOGGING=1
export CLOUDSDK_CORE_DISABLE_PROMPTS=1
export COPYFILE_DISABLE=1

PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
echo "gcloud account: ${ACCOUNT:-<none>} | project: ${PROJECT:-<none>} | zone: ${ZONE} | instance: ${INSTANCE}"

if [[ -z "${PROJECT}" || "${PROJECT}" != "${EXPECTED_PROJECT}" ]]; then
  echo "ERROR: need project ${EXPECTED_PROJECT}, got '${PROJECT:-<none>}'" >&2
  exit 1
fi

cd "${REPO}"
CODE_DIRTY=false
if ! git diff --quiet -- ml "Tema 1/results/ml_index.csv" "Tema 1/results/sweep.csv"; then
  CODE_DIRTY=true
fi
if [[ -n "$(git ls-files --others --exclude-standard -- ml)" ]]; then
  CODE_DIRTY=true
fi
GIT_COMMIT="$(git rev-parse HEAD)"
echo "GIT_COMMIT=${GIT_COMMIT} CODE_DIRTY=${CODE_DIRTY}"
printf '%s\n' "${GIT_COMMIT}" > ml/COMMIT
printf '%s\n' "${CODE_DIRTY}" > ml/CODE_DIRTY

if gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" &>/dev/null; then
  echo "Instance ${INSTANCE} already exists — reusing."
else
  echo "Creating ${INSTANCE} (${MACHINE}, ${DISK_GB}GB pd-balanced)..."
  gcloud compute instances create "${INSTANCE}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE}" \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size="${DISK_GB}GB" \
    --boot-disk-type=pd-balanced \
    --service-account="${SERVICE_ACCOUNT}" \
    --scopes=https://www.googleapis.com/auth/cloud-platform
fi

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

echo "Transferring ml + index/sweep..."
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="sudo mkdir -p /data ${REMOTE_DIR} && sudo chown -R \$(whoami): /data ${REMOTE_DIR}"

TAR="${TMPDIR:-/tmp}/tema1_ml_pre_$$.tgz"
COPYFILE_DISABLE=1 tar --disable-copyfile -C "${REPO}" -czf "${TAR}" \
  --exclude='.venv' --exclude='ml/.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='**/raw_smoke' --exclude='**/processed' --exclude='**/*.pt' \
  --exclude='Tema 1/results/_step0_inspect' --exclude='Tema 1/results/ml/runs' \
  --exclude='Tema 1/.venv' --exclude='._*' \
  ml \
  "Tema 1/results/ml_index.csv" \
  "Tema 1/results/sweep.csv" \
  2>/dev/null

gcloud compute scp "${TAR}" "${INSTANCE}:/tmp/ml_pre.tgz" --zone="${ZONE}"
rm -f "${TAR}"

gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  mkdir -p ${REMOTE_DIR}
  tar -C ${REMOTE_DIR} -xzf /tmp/ml_pre.tgz
  rm -f /tmp/ml_pre.tgz
  chmod +x ${REMOTE_DIR}/ml/cloud/run_preprocess_on_vm.sh
  # flatten: scripts expect repo root with ml/ and Tema 1/
  ls ${REMOTE_DIR}/ml ${REMOTE_DIR}/Tema\ 1/results | head
"

REMOTE_CMD="ROOT=${REMOTE_DIR} OUT=/data/processed WORKERS=${WORKERS} bash ${REMOTE_DIR}/ml/cloud/run_preprocess_on_vm.sh"

if [[ "${DETACH}" == "true" ]]; then
  echo "Starting detached preprocess (log → ${LOCAL_LOG})..."
  gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
    nohup bash -lc '${REMOTE_CMD}' > /data/logs/nohup.out 2>&1 &
    echo started_pid=\$!
    sleep 2
    tail -n 20 /data/logs/nohup.out || true
  "
  mkdir -p "$(dirname "${LOCAL_LOG}")"
  {
    echo "DETACHED $(date -Iseconds) instance=${INSTANCE} zone=${ZONE}"
    echo "GIT_COMMIT=${GIT_COMMIT} CODE_DIRTY=${CODE_DIRTY}"
    echo "Monitor: gcloud compute ssh ${INSTANCE} --zone=${ZONE} --command='tail -f /data/logs/preprocess.log'"
  } | tee "${LOCAL_LOG}"
else
  gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="${REMOTE_CMD}"
fi

echo "Launcher finished. Instance left running (DETACH=${DETACH}). Delete after verify OK."
