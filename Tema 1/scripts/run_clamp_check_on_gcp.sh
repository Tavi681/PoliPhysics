#!/usr/bin/env bash
# Clamp-check relaxed-release study on 2× on-demand c2-standard-30 + PD + GCS.
#
# Usage:
#   SHARD=0 NUM_SHARDS=2 DETACH=true bash "Tema 1/scripts/run_clamp_check_on_gcp.sh"
#   SHARD=1 NUM_SHARDS=2 DETACH=true bash "Tema 1/scripts/run_clamp_check_on_gcp.sh"
#
# Optional: PROBE_ONLY=true  — 4 jobs then stop (gate on proj wall ≤ 6 h)
#           START_RUN=false  — deploy+upload only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/.." && pwd)"
SHARD="${SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-2}"
INSTANCE="${INSTANCE:-poliphysics-tema1-clamp-s${SHARD}}"
ZONE="${ZONE:-us-central1-a}"
MACHINE="${MACHINE:-c2-standard-30}"
DETACH="${DETACH:-true}"
PROBE_ONLY="${PROBE_ONLY:-false}"
START_RUN="${START_RUN:-true}"
JOBS="${JOBS:-28}"
DISK_GB="${DISK_GB:-50}"
PD_NAME="${PD_NAME:-poliphysics-tema1-clamp-s${SHARD}-pd}"
PD_GB="${PD_GB:-100}"
EXPECTED_PROJECT="${EXPECTED_PROJECT:-authorship-verification}"
GCS_BUCKET="${GCS_BUCKET:-gs://authorship-verification-poliphysics/tema1-clampcheck}"
REMOTE_BASE="${REMOTE_BASE:-/mnt/pd/tema1}"
REMOTE_DIR="${REMOTE_BASE}/clamp"
CLAMP_RELAX_TIME="${CLAMP_RELAX_TIME:-3.0}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-ephemeral-benchmark@authorship-verification.iam.gserviceaccount.com}"
LOCAL_CLAMP="${ROOT}/results/clamp_check"

export CLOUDSDK_CORE_DISABLE_FILE_LOGGING=1
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
echo "project=${PROJECT} instance=${INSTANCE} machine=${MACHINE} shard=${SHARD}/${NUM_SHARDS}"
echo "gcs=${GCS_BUCKET} clamp_relax_time=${CLAMP_RELAX_TIME} probe_only=${PROBE_ONLY}"

if [[ -z "${PROJECT}" || "${PROJECT}" != "${EXPECTED_PROJECT}" ]]; then
  echo "ERROR: need project ${EXPECTED_PROJECT}" >&2
  exit 1
fi

if gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" &>/dev/null; then
  echo "ERROR: instance ${INSTANCE} already exists" >&2
  exit 1
fi

DEPLOY_COMMIT="${DEPLOY_COMMIT:-$(git -C "${REPO}" rev-parse HEAD)}"
echo "Deploying commit ${DEPLOY_COMMIT}"

if ! gcloud compute disks describe "${PD_NAME}" --zone="${ZONE}" &>/dev/null; then
  gcloud compute disks create "${PD_NAME}" \
    --zone="${ZONE}" --size="${PD_GB}GB" --type=pd-balanced
fi

gcloud compute instances create "${INSTANCE}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE}" \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size="${DISK_GB}GB" \
  --boot-disk-type=pd-balanced \
  --disk="name=${PD_NAME},device-name=tema1pd,mode=rw,boot=no,auto-delete=no" \
  --service-account="${SERVICE_ACCOUNT}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform

for _ in $(seq 1 60); do
  STATUS="$(gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" --format='get(status)' 2>/dev/null || true)"
  [[ "${STATUS}" == "RUNNING" ]] && break
  sleep 5
done
for _ in $(seq 1 36); do
  gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="true" 2>/dev/null && break
  sleep 5
done

STAGE=$(mktemp -d "${TMPDIR:-/tmp}/tema1_clamp_pack_XXXX")
git -C "${REPO}" archive --format=tar "${DEPLOY_COMMIT}:Tema 1" | tar -C "${STAGE}" -xf -
printf '%s\n' "${DEPLOY_COMMIT}" > "${STAGE}/COMMIT"
echo clean > "${STAGE}/COMMIT.dirty"
TAR="${TMPDIR:-/tmp}/tema1_clamp_$$.tgz"
tar -C "${STAGE}" -czf "${TAR}" \
  --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='results/ml_samples' --exclude='results/equal_t_cache' \
  --exclude='results/figs' --exclude='paper' \
  .
rm -rf "${STAGE}"

gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv python3-pip build-essential
  DEVICE=/dev/disk/by-id/google-tema1pd
  for _ in \$(seq 1 30); do [[ -e \${DEVICE} ]] && break; sleep 2; done
  if ! sudo blkid \${DEVICE} >/dev/null 2>&1; then sudo mkfs.ext4 -F \${DEVICE}; fi
  sudo mkdir -p /mnt/pd
  sudo mount \${DEVICE} /mnt/pd || true
  sudo mkdir -p ${REMOTE_DIR}
  sudo chown -R \$(whoami):\$(whoami) /mnt/pd
  UUID=\$(sudo blkid -s UUID -o value \${DEVICE} || true)
  if [[ -n \${UUID} ]] && ! grep -q '/mnt/pd' /etc/fstab; then
    echo \"UUID=\${UUID} /mnt/pd ext4 defaults,nofail 0 2\" | sudo tee -a /etc/fstab
  fi
  sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service || true
  sudo systemctl mask apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service
  sudo loginctl enable-linger \$(whoami)
"

gcloud compute scp "${TAR}" "${INSTANCE}:/tmp/tema1.tgz" --zone="${ZONE}"
rm -f "${TAR}"
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  mkdir -p ${REMOTE_DIR}
  tar -C ${REMOTE_DIR} -xzf /tmp/tema1.tgz
  rm -f /tmp/tema1.tgz
  mkdir -p ${REMOTE_DIR}/results/clamp_check
  GOT=\$(tr -d '[:space:]' < ${REMOTE_DIR}/COMMIT)
  DIRTY=\$(tr -d '[:space:]' < ${REMOTE_DIR}/COMMIT.dirty)
  echo deploy_commit=${DEPLOY_COMMIT} got=\${GOT} dirty=\${DIRTY}
  [[ \${GOT} == ${DEPLOY_COMMIT} ]] || { echo ABORT_commit; exit 3; }
  [[ \${DIRTY} == clean ]] || { echo ABORT_dirty; exit 3; }
"

# Upload Mac completed clamp_check CSVs for resume skip
if [[ -d "${LOCAL_CLAMP}" ]]; then
  echo "Uploading local clamp_check CSVs for resume..."
  gsutil -m rsync -r -x '.*\.(npz|log|json|md)$' \
    "${LOCAL_CLAMP}/" "${GCS_BUCKET}/shared/" || true
  gcloud compute scp --recurse \
    "${LOCAL_CLAMP}/relaxed_release_sweep.csv" \
    "${INSTANCE}:${REMOTE_DIR}/results/clamp_check/" --zone="${ZONE}" || true
fi

# Ensure GCS shard prefix is writable
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  echo gate | gsutil cp - ${GCS_BUCKET}/shard-${SHARD}/_write_test.txt
"

gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  export PYTHONUNBUFFERED=1
  cd ${REMOTE_DIR}
  python3 -m venv ${REMOTE_BASE}/venv
  source ${REMOTE_BASE}/venv/bin/activate
  pip install -U pip wheel
  pip install -q -e ${REMOTE_DIR}
  python - <<'PY'
from ballooning.io_hdf5 import git_provenance
p = git_provenance()
print('git_provenance', p)
if p['commit'] != '${DEPLOY_COMMIT}' or p['dirty']:
    raise SystemExit('ABORT: provenance')
print('GATE_OK')
PY
"

if [[ "${START_RUN}" != "true" ]]; then
  echo "START_RUN=false: deploy only"
  exit 0
fi

# Probe gate
echo "Running 4-job probe on shard ${SHARD}..."
set +e
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  export PYTHONUNBUFFERED=1
  cd ${REMOTE_DIR}
  source ${REMOTE_BASE}/venv/bin/activate
  python -u scripts/clamp_check_run.py --probe \
    --shard ${SHARD} --num-shards ${NUM_SHARDS} --workers ${JOBS} \
    --clamp-relax-time ${CLAMP_RELAX_TIME} --max-proj-h 6.0 \
    | tee ${REMOTE_DIR}/results/clamp_check/probe_shard_${SHARD}.log
"
PROBE_EC=$?
set -e
if [[ ${PROBE_EC} -ne 0 ]]; then
  echo "PROBE failed or projected > 6 h (exit=${PROBE_EC}). Not starting full run."
  gcloud compute scp \
    "${INSTANCE}:${REMOTE_DIR}/results/clamp_check/probe_shard_${SHARD}.log" \
    "${LOCAL_CLAMP}/" --zone="${ZONE}" || true
  exit ${PROBE_EC}
fi

if [[ "${PROBE_ONLY}" == "true" ]]; then
  echo "PROBE_ONLY=true: stopping after probe"
  exit 0
fi

echo "Starting full clamp_check as systemd tema1-clampcheck..."
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  cat > ${REMOTE_BASE}/run_clamp.sh <<'EOS'
#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
cd ${REMOTE_DIR}
source ${REMOTE_BASE}/venv/bin/activate
(
  while true; do
    sleep 600
    gsutil -m rsync -r ${REMOTE_DIR}/results/clamp_check ${GCS_BUCKET}/shard-${SHARD}/ || true
  done
) >/dev/null 2>&1 &
echo \$! > ${REMOTE_BASE}/rsync.pid
exec python -u scripts/clamp_check_run.py --run \
  --shard ${SHARD} --num-shards ${NUM_SHARDS} --workers ${JOBS} \
  --clamp-relax-time ${CLAMP_RELAX_TIME}
EOS
  # Expand remote vars into the script (double-quoted heredoc via SSH already done above — fix):
  sed -i \"s|\\\${REMOTE_DIR}|${REMOTE_DIR}|g; s|\\\${REMOTE_BASE}|${REMOTE_BASE}|g; s|\\\${GCS_BUCKET}|${GCS_BUCKET}|g; s|\\\${SHARD}|${SHARD}|g; s|\\\${NUM_SHARDS}|${NUM_SHARDS}|g; s|\\\${JOBS}|${JOBS}|g; s|\\\${CLAMP_RELAX_TIME}|${CLAMP_RELAX_TIME}|g\" ${REMOTE_BASE}/run_clamp.sh || true
  chmod +x ${REMOTE_BASE}/run_clamp.sh
  # Rewrite cleanly
  cat > ${REMOTE_BASE}/run_clamp.sh <<EOF
#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
cd ${REMOTE_DIR}
source ${REMOTE_BASE}/venv/bin/activate
(
  while true; do
    sleep 600
    gsutil -m rsync -r ${REMOTE_DIR}/results/clamp_check ${GCS_BUCKET}/shard-${SHARD}/ || true
  done
) >/dev/null 2>&1 &
echo \\\$! > ${REMOTE_BASE}/rsync.pid
exec python -u scripts/clamp_check_run.py --run \\
  --shard ${SHARD} --num-shards ${NUM_SHARDS} --workers ${JOBS} \\
  --clamp-relax-time ${CLAMP_RELAX_TIME}
EOF
  chmod +x ${REMOTE_BASE}/run_clamp.sh
  sudo tee /etc/systemd/system/tema1-clampcheck.service >/dev/null <<EOF
[Unit]
Description=Tema 1 clamp_check relaxed release
After=network-online.target
Wants=network-online.target
RequiresMountsFor=/mnt/pd

[Service]
Type=simple
User=\$(whoami)
Group=\$(whoami)
WorkingDirectory=${REMOTE_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=${REMOTE_BASE}/run_clamp.sh
Restart=no
KillMode=mixed
TimeoutStopSec=60
StandardOutput=append:${REMOTE_BASE}/clampcheck.log
StandardError=append:${REMOTE_BASE}/clampcheck.log

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable --now tema1-clampcheck.service
  echo SYSTEMD=tema1-clampcheck SHARD=${SHARD}
"

echo "Launched ${INSTANCE}. Log: ${REMOTE_BASE}/clampcheck.log"
if [[ "${DETACH}" == "true" ]]; then
  exit 0
fi
