#!/usr/bin/env bash
# B1 + B2 shards on on-demand c2-standard-32 + persistent disk + GCS rsync.
# Do NOT run until R and B0 both PASS and you confirm.
#
# Usage (after confirmation):
#   SHARD=0 NUM_SHARDS=2 DETACH=true bash "Tema 1/scripts/run_stage_b12_on_gcp.sh"
#   SHARD=1 NUM_SHARDS=2 DETACH=true bash "Tema 1/scripts/run_stage_b12_on_gcp.sh"
#
# Optional:
#   CALIBRATE=true   — 2 short runs / N, write calibrate.json, then continue
#   GCS_BUCKET=gs://authorship-verification-poliphysics/tema1-stageb
#   INSTANCE / ZONE / JOBS / PD_NAME
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/.." && pwd)"
SHARD="${SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
INSTANCE="${INSTANCE:-poliphysics-tema1-b12-s${SHARD}}"
ZONE="${ZONE:-us-central1-a}"
MACHINE="${MACHINE:-c2-standard-32}"
PREEMPTIBLE=false
DETACH="${DETACH:-true}"
CALIBRATE="${CALIBRATE:-false}"
START_SWEEP="${START_SWEEP:-true}"
JOBS="${JOBS:-32}"
DISK_GB="${DISK_GB:-50}"
PD_NAME="${PD_NAME:-poliphysics-tema1-b12-s${SHARD}-pd}"
PD_GB="${PD_GB:-300}"
EXPECTED_PROJECT="${EXPECTED_PROJECT:-authorship-verification}"
GCS_BUCKET="${GCS_BUCKET:-gs://authorship-verification-poliphysics/tema1-stageb}"
REMOTE_BASE="${REMOTE_BASE:-/mnt/pd/tema1}"
REMOTE_DIR="${REMOTE_BASE}/new"
LOCAL_RESULTS="${LOCAL_RESULTS:-${ROOT}/results}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-ephemeral-benchmark@authorship-verification.iam.gserviceaccount.com}"

export CLOUDSDK_CORE_DISABLE_FILE_LOGGING=1
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
echo "gcloud account: ${ACCOUNT:-<none>} | project: ${PROJECT:-<none>}"
echo "instance=${INSTANCE} machine=${MACHINE} preemptible=${PREEMPTIBLE}"
echo "shard=${SHARD}/${NUM_SHARDS} workers=${JOBS} calibrate=${CALIBRATE}"
echo "gcs=${GCS_BUCKET} pd=${PD_NAME}"

if [[ -z "${PROJECT}" || "${PROJECT}" != "${EXPECTED_PROJECT}" ]]; then
  echo "ERROR: need project ${EXPECTED_PROJECT}, got '${PROJECT:-<none>}'" >&2
  exit 1
fi

if [[ "${PREEMPTIBLE}" != "false" ]]; then
  echo "ERROR: Spot/preemptible is forbidden for Stage B." >&2
  exit 1
fi

if gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" &>/dev/null; then
  echo "ERROR: instance ${INSTANCE} already exists — refuse to clobber." >&2
  exit 1
fi

TAR=""
DELETE_ON_EXIT=0
cleanup() {
  local ec=$?
  rm -f "${TAR}"
  if [[ "${DELETE_ON_EXIT}" == "1" ]]; then
    echo "Deleting instance ${INSTANCE} (PD ${PD_NAME} kept)..."
    gcloud compute instances delete "${INSTANCE}" --zone="${ZONE}" --quiet || true
  else
    echo "Leaving instance ${INSTANCE} running. PD=${PD_NAME} persists."
  fi
  return "${ec}"
}
trap cleanup EXIT

if ! gcloud compute disks describe "${PD_NAME}" --zone="${ZONE}" &>/dev/null; then
  echo "Creating persistent disk ${PD_NAME} (${PD_GB}GB)..."
  gcloud compute disks create "${PD_NAME}" \
    --zone="${ZONE}" --size="${PD_GB}GB" --type=pd-balanced
fi

CREATE_ARGS=(
  --zone="${ZONE}"
  --machine-type="${MACHINE}"
  --image-family=ubuntu-2404-lts-amd64
  --image-project=ubuntu-os-cloud
  --boot-disk-size="${DISK_GB}GB"
  --boot-disk-type=pd-balanced
  --disk="name=${PD_NAME},device-name=tema1pd,mode=rw,boot=no,auto-delete=no"
  --service-account="${SERVICE_ACCOUNT}"
  --scopes=https://www.googleapis.com/auth/cloud-platform
)

echo "Creating ${INSTANCE} (on-demand, no Spot)..."
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

DEPLOY_COMMIT="${DEPLOY_COMMIT:-$(git -C "${REPO}" rev-parse HEAD)}"
if [[ -z "${DEPLOY_COMMIT}" || "${DEPLOY_COMMIT}" == "unknown" ]]; then
  echo "ERROR: cannot resolve DEPLOY_COMMIT" >&2
  exit 1
fi
echo "Deploying exact commit ${DEPLOY_COMMIT} (git archive, dirty=clean)"
# Pack the committed tree only — never the dirty working directory.
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/tema1_b12_pack_XXXX")
git -C "${REPO}" archive --format=tar "${DEPLOY_COMMIT}:Tema 1" | tar -C "${STAGE}" -xf -
printf '%s\n' "${DEPLOY_COMMIT}" > "${STAGE}/COMMIT"
echo clean > "${STAGE}/COMMIT.dirty"
TAR="${TMPDIR:-/tmp}/tema1_b12_$$.tgz"
tar -C "${STAGE}" -czf "${TAR}" \
  --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='results/ml_samples' --exclude='results/ml_samples_smoke' \
  --exclude='results/equal_t_cache' --exclude='results/figs' \
  --exclude='paper' \
  .
rm -rf "${STAGE}"
echo "${DEPLOY_COMMIT}" > "${ROOT}/COMMIT"
echo clean > "${ROOT}/COMMIT.dirty"

gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv python3-pip build-essential tmux
  DEVICE=/dev/disk/by-id/google-tema1pd
  for _ in \$(seq 1 30); do
    [[ -e \${DEVICE} ]] && break
    sleep 2
  done
  if ! sudo blkid \${DEVICE} >/dev/null 2>&1; then
    sudo mkfs.ext4 -F \${DEVICE}
  fi
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
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  mkdir -p ${REMOTE_DIR}
  tar -C ${REMOTE_DIR} -xzf /tmp/tema1.tgz
  rm -f /tmp/tema1.tgz
  mkdir -p ${REMOTE_DIR}/results/ml_samples ${REMOTE_DIR}/results/snapshots
  GOT=\$(tr -d '[:space:]' < ${REMOTE_DIR}/COMMIT)
  DIRTY=\$(tr -d '[:space:]' < ${REMOTE_DIR}/COMMIT.dirty)
  echo deploy_commit=${DEPLOY_COMMIT} got=\${GOT} dirty=\${DIRTY}
  if [[ \${GOT} != ${DEPLOY_COMMIT} ]]; then
    echo 'ABORT: commit hash mismatch on VM' >&2
    exit 3
  fi
  if [[ \${DIRTY} != clean ]]; then
    echo 'ABORT: dirty!=False on VM' >&2
    exit 3
  fi
"

echo "Checking GCS prefix ${GCS_BUCKET} (object write; bucket create is done locally)..."
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  echo gate-write | gsutil cp - ${GCS_BUCKET}/_write_test_${SHARD}.txt
"

CAL_CMD=""
if [[ "${CALIBRATE}" == "true" ]]; then
  CAL_CMD="python -u scripts/stage_b.py --calibrate --workers ${JOBS}; "
fi

echo "Installing venv and starting shard ${SHARD}/${NUM_SHARDS} as systemd unit tema1-sweep..."
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
open('${REMOTE_BASE}/deploy_gate.txt', 'w').write(
    f\"commit={p['commit']}\\ndirty={p['dirty']}\\nsource={p['source']}\\n\")
if p['commit'] != '${DEPLOY_COMMIT}' or p['dirty']:
    raise SystemExit('ABORT: git_provenance dirty or hash mismatch')
print('GATE_OK dirty=False commit=${DEPLOY_COMMIT}')
PY
  if [[ '${START_SWEEP:-true}' != 'true' ]]; then
    echo 'START_SWEEP=false: deploy+gate only, not starting sweep'
    exit 0
  fi
  : > ${REMOTE_BASE}/stageb.log
  tmux kill-session -t stageb 2>/dev/null || true
  cat > ${REMOTE_BASE}/run_sweep.sh <<'EOS'
#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
cd ${REMOTE_DIR}
source ${REMOTE_BASE}/venv/bin/activate
${CAL_CMD}
(
  while true; do
    sleep 600
    gsutil -m rsync -r ${REMOTE_DIR}/results ${GCS_BUCKET}/shard-${SHARD}/ || true
  done
) >/dev/null 2>&1 &
echo \$! > ${REMOTE_BASE}/rsync.pid
exec python -u scripts/stage_b.py --sweep --workers ${JOBS} \\
  --shard ${SHARD} --num-shards ${NUM_SHARDS}
EOS
  # The heredoc above is expanded locally via the SSH double-quoted command.
  chmod +x ${REMOTE_BASE}/run_sweep.sh
  sudo tee /etc/systemd/system/tema1-sweep.service >/dev/null <<EOF
[Unit]
Description=Tema 1 B1/B2 production sweep
After=network-online.target
Wants=network-online.target
RequiresMountsFor=/mnt/pd

[Service]
Type=simple
User=\$(whoami)
Group=\$(whoami)
WorkingDirectory=${REMOTE_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=${REMOTE_BASE}/run_sweep.sh
Restart=no
KillMode=mixed
TimeoutStopSec=60
StandardOutput=append:${REMOTE_BASE}/stageb.log
StandardError=append:${REMOTE_BASE}/stageb.log

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable --now tema1-sweep.service
  echo SYSTEMD=tema1-sweep SHARD=${SHARD}/${NUM_SHARDS}
"

echo "DETACH=${DETACH}: B1/B2 shard ${SHARD} running on ${INSTANCE}."
echo "Status: INSTANCE=${INSTANCE} bash \"${ROOT}/scripts/check_stage_b_gcp.sh\""
echo "GCS rsync every 10 min → ${GCS_BUCKET}/shard-${SHARD}/"
echo "Merge after all shards finish:"
echo "  python scripts/merge_sweep_shards.py"
exit 0
