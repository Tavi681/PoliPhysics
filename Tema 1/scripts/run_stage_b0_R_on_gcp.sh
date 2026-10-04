#!/usr/bin/env bash
# R + B0 on one on-demand c2-standard-16 + persistent disk.
# Runs regression (R1–R3) and B0 (T1–T6) in parallel under tmux.
# PREEMPTIBLE is forced false. Does NOT launch B1/B2.
#
# Usage:
#   DETACH=true bash "Tema 1/scripts/run_stage_b0_R_on_gcp.sh"
#
# After both finish, pull reports locally and wait for confirmation
# before delete / B1 launch:
#   bash "Tema 1/scripts/check_stage_b_gcp.sh"
#   INSTANCE=poliphysics-tema1-b0 bash "Tema 1/scripts/run_stage_b0_R_on_gcp.sh" pull
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # Tema 1/
REPO="$(cd "${ROOT}/.." && pwd)"                         # PoliPhysics/
INSTANCE="${INSTANCE:-poliphysics-tema1-b0}"
ZONE="${ZONE:-us-central1-a}"
MACHINE="${MACHINE:-c2-standard-16}"
PREEMPTIBLE=false
DETACH="${DETACH:-true}"
JOBS="${JOBS:-16}"
B0_WORKERS="${B0_WORKERS:-8}"
R_WORKERS_EACH="${R_WORKERS_EACH:-4}"
DISK_GB="${DISK_GB:-50}"
PD_NAME="${PD_NAME:-poliphysics-tema1-pd}"
PD_GB="${PD_GB:-200}"
EXPECTED_PROJECT="${EXPECTED_PROJECT:-authorship-verification}"
REMOTE_BASE="${REMOTE_BASE:-/mnt/pd/tema1}"
REMOTE_NEW="${REMOTE_BASE}/new"
REMOTE_OLD="${REMOTE_BASE}/old"
LOCAL_RESULTS="${LOCAL_RESULTS:-${ROOT}/results}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-ephemeral-benchmark@authorship-verification.iam.gserviceaccount.com}"
OLD_COMMIT="${OLD_COMMIT:-4ae88b1}"

export CLOUDSDK_CORE_DISABLE_FILE_LOGGING=1
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

if [[ "${1:-}" == "pull" ]]; then
  mkdir -p "${LOCAL_RESULTS}"
  for f in regression.txt b0_report.json b0_report.txt b0_estimate.txt \
           b0_config.json fig_ksvalid.csv fig_ksvalid_pdf.csv tab_wc.csv \
           B0_DONE R_DONE meta.json; do
    gcloud compute scp \
      "${INSTANCE}:${REMOTE_NEW}/results/${f}" \
      "${LOCAL_RESULTS}/" --zone="${ZONE}" 2>/dev/null || true
  done
  gcloud compute scp \
    "${INSTANCE}:${REMOTE_BASE}/stageb.log" \
    "${LOCAL_RESULTS}/stage_b0_R_gcp.log" --zone="${ZONE}" 2>/dev/null || true
  echo "Pulled reports into ${LOCAL_RESULTS}."
  echo "VM ${INSTANCE} left running. Confirm before delete or B1 launch."
  exit 0
fi

PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
echo "gcloud account: ${ACCOUNT:-<none>} | project: ${PROJECT:-<none>} | zone: ${ZONE} | instance: ${INSTANCE}"
echo "machine: ${MACHINE} preemptible=${PREEMPTIBLE} detach=${DETACH} jobs=${JOBS}"
echo "pd=${PD_NAME} (${PD_GB}GB)  B0_workers=${B0_WORKERS}  R_workers_each=${R_WORKERS_EACH}"

if [[ -z "${PROJECT}" || "${PROJECT}" != "${EXPECTED_PROJECT}" ]]; then
  echo "ERROR: need project ${EXPECTED_PROJECT}, got '${PROJECT:-<none>}'" >&2
  exit 1
fi

if gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" &>/dev/null; then
  echo "ERROR: instance ${INSTANCE} already exists — refuse to clobber." >&2
  echo "Use INSTANCE=... or delete it. For pull: $0 pull" >&2
  exit 1
fi

TAR=""
DELETE_ON_EXIT=0
if [[ "${DETACH}" != "true" ]]; then
  DELETE_ON_EXIT=1
fi
cleanup() {
  local ec=$?
  rm -f "${TAR}" "${TAR}.old.tgz"
  if [[ "${DELETE_ON_EXIT}" == "1" ]]; then
    echo "Deleting instance ${INSTANCE} (PD ${PD_NAME} is kept)..."
    gcloud compute instances delete "${INSTANCE}" --zone="${ZONE}" --quiet || true
  else
    echo "Leaving instance ${INSTANCE} running (DETACH/no-delete). PD=${PD_NAME} persists."
  fi
  return "${ec}"
}
trap cleanup EXIT

if ! gcloud compute disks describe "${PD_NAME}" --zone="${ZONE}" &>/dev/null; then
  echo "Creating persistent disk ${PD_NAME} (${PD_GB}GB)..."
  gcloud compute disks create "${PD_NAME}" \
    --zone="${ZONE}" --size="${PD_GB}GB" --type=pd-balanced
else
  echo "Reusing persistent disk ${PD_NAME}"
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

git -C "${REPO}" rev-parse HEAD > "${ROOT}/COMMIT" 2>/dev/null || echo "unknown" > "${ROOT}/COMMIT"
if git -C "${REPO}" status --porcelain 2>/dev/null | grep -q .; then
  echo dirty > "${ROOT}/COMMIT.dirty"
else
  echo clean > "${ROOT}/COMMIT.dirty"
fi
echo "Packed commit (new): $(cat "${ROOT}/COMMIT") dirty=$(cat "${ROOT}/COMMIT.dirty")"

TAR="${TMPDIR:-/tmp}/tema1_b0_$$.tgz"
tar -C "${ROOT}" -czf "${TAR}" \
  --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='.cache' --exclude='*.egg-info' \
  --exclude='results/ml_samples' --exclude='results/ml_samples_smoke' \
  --exclude='results/equal_t_cache' --exclude='results/equal_t_cache_smoke' \
  --exclude='results/figs' --exclude='results/pre_stageA' \
  --exclude='paper' \
  ballooning scripts tests pyproject.toml README.md COMMIT COMMIT.dirty \
  baseline_mac results

OLD_TAR="${TAR}.old.tgz"
git -C "${REPO}" archive --format=tar "${OLD_COMMIT}:Tema 1" | gzip > "${OLD_TAR}"

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
  sudo mkdir -p ${REMOTE_NEW} ${REMOTE_OLD} ${REMOTE_BASE}
  sudo chown -R \$(whoami):\$(whoami) /mnt/pd
"

gcloud compute scp "${TAR}" "${INSTANCE}:/tmp/tema1_new.tgz" --zone="${ZONE}"
gcloud compute scp "${OLD_TAR}" "${INSTANCE}:/tmp/tema1_old.tgz" --zone="${ZONE}"
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  mkdir -p ${REMOTE_NEW} ${REMOTE_OLD}
  tar -C ${REMOTE_NEW} -xzf /tmp/tema1_new.tgz
  tar -C ${REMOTE_OLD} -xzf /tmp/tema1_old.tgz
  rm -f /tmp/tema1_new.tgz /tmp/tema1_old.tgz
  mkdir -p ${REMOTE_NEW}/results ${REMOTE_OLD}/results
"

echo "Installing venv and starting R || B0 under tmux session 'stageb'..."
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  set -euo pipefail
  export PYTHONUNBUFFERED=1
  cd ${REMOTE_NEW}
  python3 -m venv ${REMOTE_BASE}/venv
  source ${REMOTE_BASE}/venv/bin/activate
  pip install -U pip wheel
  pip install -q -e ${REMOTE_NEW}
  : > ${REMOTE_BASE}/stageb.log
  rm -f ${REMOTE_NEW}/results/B0_DONE ${REMOTE_NEW}/results/R_DONE
  tmux kill-session -t stageb 2>/dev/null || true
  tmux new -d -s stageb -n wrap
  tmux send-keys -t stageb:wrap \"
    set -euo pipefail
    source ${REMOTE_BASE}/venv/bin/activate
    export PYTHONUNBUFFERED=1
    cd ${REMOTE_NEW}
    python -u scripts/stage_b.py --b0 --workers ${B0_WORKERS} > ${REMOTE_NEW}/results/b0.log 2>&1 &
    echo \\\$! > ${REMOTE_BASE}/b0.pid
    python -u scripts/run_regression_cloud.py \\
      --old-dir ${REMOTE_OLD} --new-dir ${REMOTE_NEW} \\
      --baseline-mac ${REMOTE_NEW}/baseline_mac \\
      --workers-each ${R_WORKERS_EACH} \\
      --out ${REMOTE_NEW}/results/regression.txt > ${REMOTE_NEW}/results/r.log 2>&1 &
    echo \\\$! > ${REMOTE_BASE}/r.pid
    echo B0_PID=\\\$(cat ${REMOTE_BASE}/b0.pid) R_PID=\\\$(cat ${REMOTE_BASE}/r.pid)
    wait || true
    echo ALL_CHILDREN_DONE
  \" C-m
  echo TMUX=stageb
"

echo "DETACH=${DETACH}: R and B0 running on ${INSTANCE} under tmux 'stageb'."
echo "Status:"
echo "  INSTANCE=${INSTANCE} bash \"${ROOT}/scripts/check_stage_b_gcp.sh\""
echo "Pull reports when B0_DONE and R_DONE exist:"
echo "  INSTANCE=${INSTANCE} bash \"${ROOT}/scripts/run_stage_b0_R_on_gcp.sh\" pull"
echo "STOP. Do not launch B1/B2 until both reports PASS and you confirm."
echo "PD ${PD_NAME} survives VM delete. Leave the VM up until you confirm."
exit 0
