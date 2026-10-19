#!/usr/bin/env bash
# Create g2-standard-8 + 1x L4 (us-central1), sync code, launch run_train_on_vm.sh.
# Does NOT delete the VM. Does NOT delete b0a7c5b503af5d26. Does NOT run the k-sweep.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTANCE="${INSTANCE:-poliphysics-tema1-ml-train}"
ZONE="${ZONE:-us-central1-b}"
MACHINE="${MACHINE:-g2-standard-8}"
DISK_GB="${DISK_GB:-200}"
EXPECTED_PROJECT="${EXPECTED_PROJECT:-authorship-verification}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-ephemeral-benchmark@authorship-verification.iam.gserviceaccount.com}"
REMOTE_DIR="${REMOTE_DIR:-/home/octav/poliphysics}"
LOCAL_LOG="${LOCAL_LOG:-${REPO}/results/ml/train_gcp.log}"

export CLOUDSDK_CORE_DISABLE_FILE_LOGGING=1
export CLOUDSDK_CORE_DISABLE_PROMPTS=1
export COPYFILE_DISABLE=1

PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
echo "project=${PROJECT:-<none>} zone=${ZONE} instance=${INSTANCE}"
if [[ "${PROJECT}" != "${EXPECTED_PROJECT}" ]]; then
  echo "ERROR: need project ${EXPECTED_PROJECT}" >&2
  exit 1
fi

cd "${REPO}"
GIT_COMMIT="$(git rev-parse HEAD)"
echo "GIT_COMMIT=${GIT_COMMIT}"
printf '%s\n' "${GIT_COMMIT}" > ml/COMMIT
printf 'false\n' > ml/CODE_DIRTY

if ! gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" &>/dev/null; then
  echo "Creating ${INSTANCE} (${MACHINE} + nvidia-l4)..."
  gcloud compute instances create "${INSTANCE}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE}" \
    --accelerator=type=nvidia-l4,count=1 \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size="${DISK_GB}GB" \
    --boot-disk-type=pd-balanced \
    --service-account="${SERVICE_ACCOUNT}" \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata=install-nvidia-driver=True
else
  STATUS="$(gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" --format='get(status)' 2>/dev/null || true)"
  if [[ "${STATUS}" != "RUNNING" ]]; then
    echo "Starting existing ${INSTANCE} (status=${STATUS}) — no resize/recreate/delete..."
    gcloud compute instances start "${INSTANCE}" --zone="${ZONE}"
  fi
fi

for _ in $(seq 1 90); do
  STATUS="$(gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" --format='get(status)' 2>/dev/null || true)"
  [[ "${STATUS}" == "RUNNING" ]] && break
  sleep 5
done
for _ in $(seq 1 60); do
  gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="true" 2>/dev/null && break
  sleep 5
done

gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="sudo mkdir -p /data ${REMOTE_DIR} && sudo chown -R \$(whoami): /data ${REMOTE_DIR}"
TAR="${TMPDIR:-/tmp}/tema1_ml_train_$$.tgz"
COPYFILE_DISABLE=1 tar --disable-copyfile -C "${REPO}" -czf "${TAR}" \
  --exclude='.venv' --exclude='ml/.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='**/processed' --exclude='**/*.pt' --exclude='results/ml/runs' \
  --exclude='._*' \
  ml "ballooning" "results/ml/splits"
gcloud compute scp "${TAR}" "${INSTANCE}:/tmp/tema1_ml.tgz" --zone="${ZONE}"
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="mkdir -p ${REMOTE_DIR}/Tema\ 1 && tar -xzf /tmp/tema1_ml.tgz -C ${REMOTE_DIR}/Tema\ 1 && chmod +x ${REMOTE_DIR}/Tema\ 1/ml/cloud/run_train_on_vm.sh"

# Optional overrides (defaults keep full local→baseline pipeline).
CONFIGS="${CONFIGS:-ml/configs/gns_local_gpu.yaml ml/configs/gns_baseline_gpu.yaml}"
COST_CAP_USD="${COST_CAP_USD:-40}"
TRAIN_SESSION_T0="${TRAIN_SESSION_T0:-}"

echo "Launching remote train (DETACH) CONFIGS=${CONFIGS} COST_CAP_USD=${COST_CAP_USD}..."
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  mkdir -p ${REMOTE_DIR}/Tema\\ 1/results/ml/logs
  rm -f ${REMOTE_DIR}/Tema\\ 1/results/ml/logs/sanity_fail.code
  rm -f ${REMOTE_DIR}/Tema\\ 1/results/ml/logs/sanity_error.code
  nohup env REMOTE_DIR=${REMOTE_DIR} DATA_VERSION=5c60fa669a719404 \
    CONFIGS='${CONFIGS}' COST_CAP_USD='${COST_CAP_USD}' \
    TRAIN_SESSION_T0='${TRAIN_SESSION_T0}' \
    bash ${REMOTE_DIR}/Tema\ 1/ml/cloud/run_train_on_vm.sh \
    > ${REMOTE_DIR}/Tema\\ 1/results/ml/logs/train_vm.log 2>&1 < /dev/null &
  echo STARTED_PID=\$!
" | tee "${LOCAL_LOG}"

echo "Train launched on ${INSTANCE}. VM will NOT be deleted. Log: ${LOCAL_LOG}"
