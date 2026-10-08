#!/usr/bin/env bash
# Mac-side Round-8c supervisor: quota → 4 parallel shards → pull+delete each.
# No wall-time cap. Always delete the VM that this script created.
# SSH probes are hard-timeouted so a hung tail cannot block pull of another shard.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/.." && pwd)"
ZONE="${ZONE:-us-central1-a}"
PROJECT="${PROJECT:-authorship-verification}"
MACHINE_VCPU=16
LAUNCH="${ROOT}/scripts/run_round8c_export_on_gcp.sh"
LOG="${ROOT}/paper_results/round8c_supervisor.log"
export COPYFILE_DISABLE=1
mkdir -p "${ROOT}/paper_results"
exec > >(tee -a "${LOG}") 2>&1

echo "=== round8c supervisor start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
cd "${REPO}"

python3 - <<'PY'
import json, subprocess
raw = subprocess.check_output(
    ["gcloud", "compute", "regions", "describe", "us-central1",
     "--project=authorship-verification", "--format=json"])
d = json.loads(raw)
for q in d.get("quotas", []):
    m = q.get("metric", "")
    if m in ("C2_CPUS", "CPUS"):
        print(f"quota {m}: limit={q.get('limit')} usage={q.get('usage')}")
PY
gcloud compute instances list --project="${PROJECT}" || true

C2_LIMIT=$(gcloud compute regions describe us-central1 --project="${PROJECT}" \
  --format='json' | python3 -c "import sys,json; d=json.load(sys.stdin);
print(next(q['limit'] for q in d['quotas'] if q['metric']=='C2_CPUS'))")
C2_USE=$(gcloud compute regions describe us-central1 --project="${PROJECT}" \
  --format='json' | python3 -c "import sys,json; d=json.load(sys.stdin);
print(next(q['usage'] for q in d['quotas'] if q['metric']=='C2_CPUS'))")
FREE=$(python3 -c "print(max(0, int(float(${C2_LIMIT})) - int(float(${C2_USE}))))")
# Tema 1 C2 is in europe-west4; us-central1 leftover is 100. Do not shrink
# MAX_PAR when our own r8c VMs are already counted in usage.
MAX_PAR=$(python3 -c "print(max(1, min(4, int(float(${C2_LIMIT}))//${MACHINE_VCPU})))")
echo "C2 limit=${C2_LIMIT} usage=${C2_USE} free=${FREE} max parallel=${MAX_PAR}"
if (( MAX_PAR < 4 )); then
  echo "Quota does not allow all 4 shards at once; queueing (cap ${MAX_PAR})."
fi

inst_name() {
  printf 'poliphysics-tema2-r8c-%s' "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
}

ssh_cmd() {
  local inst="$1"
  shift
  local to="${SSH_TIMEOUT:-25}"
  python3 - <<PY
import subprocess, sys
cmd = ["gcloud", "compute", "ssh", "${inst}", "--zone=${ZONE}",
       "--ssh-flag=-o ConnectTimeout=10",
       "--ssh-flag=-o ServerAliveInterval=10",
       "--ssh-flag=-o ServerAliveCountMax=3",
       "--command=" + """$*"""]
try:
    r = subprocess.run(cmd, timeout=int("${to}"), capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    sys.exit(r.returncode)
except subprocess.TimeoutExpired:
    print("ssh timed out", file=sys.stderr)
    sys.exit(124)
PY
}

launch_shard() {
  local shard="$1"
  local jobs="${2:-16}"
  local inst
  inst="$(inst_name "${shard}")"
  if gcloud compute instances describe "${inst}" --zone="${ZONE}" &>/dev/null; then
    echo "Instance ${inst} already exists — skip create, will wait."
    return 0
  fi
  echo "Launching shard ${shard} JOBS=${jobs}..."
  DETACH=true SHARD="${shard}" JOBS="${jobs}" bash "${LAUNCH}"
}

shard_done() {
  local shard="$1"
  local inst
  inst="$(inst_name "${shard}")"
  ssh_cmd "${inst}" "test -f /home/octav/tema2/_ROUND8C_${shard}_DONE"
}

pull_delete() {
  local shard="$1"
  local inst
  inst="$(inst_name "${shard}")"
  echo "Pull + delete ${inst}..."
  mkdir -p "${ROOT}/paper_results/round8c/${shard}"
  SSH_TIMEOUT=90 ssh_cmd "${inst}" "tar -C /home/octav/tema2 -czf /tmp/tema2_r8c.tgz paper_results/round8c/${shard} round8c_${shard}.log" || true
  gcloud compute scp "${inst}:/tmp/tema2_r8c.tgz" \
    "/tmp/tema2_r8c_${shard}.tgz" --zone="${ZONE}" || true
  python3 - <<PY
from pathlib import Path
import tarfile
p = Path("/tmp/tema2_r8c_${shard}.tgz")
root = Path(r"${ROOT}")
if p.is_file():
    with tarfile.open(p) as t:
        t.extractall(root)
        print("extracted", p.name, "members", len(t.getnames()))
    njson = len(list((root/"paper_results/round8c"/"${shard}"/"runs").glob("*.json"))) if (root/"paper_results/round8c"/"${shard}"/"runs").exists() else 0
    print("row json files", njson)
PY
  if [[ -f "${ROOT}/round8c_${shard}.log" ]]; then
    cp "${ROOT}/round8c_${shard}.log" "${ROOT}/paper_results/round8c_${shard}_gcp.log"
  fi
  echo "Deleting ${inst} now."
  gcloud compute instances delete "${inst}" --zone="${ZONE}" --quiet
  gcloud compute instances list --project="${PROJECT}" || true
}

JOBS="${JOBS:-16}"
echo "Using JOBS=${JOBS}"

echo "=== local job plan ==="
cd "${ROOT}"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="${ROOT}/.venv/bin/python3"
fi
"${PY}" -u -m validation.round8c --plan || true
cd "${REPO}"

QUEUE=(A B C D)
declare -a RUNNING=()
LAST_PROGRESS=0

still_needed() {
  local s="$1"
  [[ ! -f "${ROOT}/paper_results/round8c/${s}/.pulled" ]]
}

mark_pulled() {
  mkdir -p "${ROOT}/paper_results/round8c/$1"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${ROOT}/paper_results/round8c/$1/.pulled"
}

while true; do
  NEW_RUN=()
  for s in "${RUNNING[@]+"${RUNNING[@]}"}"; do
    if shard_done "${s}"; then
      pull_delete "${s}"
      mark_pulled "${s}"
    else
      echo "[sup] ${s} running $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      ssh_cmd "$(inst_name "${s}")" "tail -n 4 /home/octav/tema2/round8c_${s}.log" || true
      NEW_RUN+=("${s}")
    fi
  done
  RUNNING=("${NEW_RUN[@]+"${NEW_RUN[@]}"}")

  while (( ${#RUNNING[@]} < MAX_PAR )) && (( ${#QUEUE[@]} > 0 )); do
    nxt="${QUEUE[0]}"
    QUEUE=("${QUEUE[@]:1}")
    if ! still_needed "${nxt}"; then
      echo "skip ${nxt}, already pulled"
      continue
    fi
    launch_shard "${nxt}" "${JOBS}"
    RUNNING+=("${nxt}")
  done

  if (( ${#RUNNING[@]} == 0 && ${#QUEUE[@]} == 0 )); then
    break
  fi
  NOW=$(date +%s)
  if (( NOW - LAST_PROGRESS >= 1800 )); then
    echo "[sup] 30 min tick $(date -u +%Y-%m-%dT%H:%M:%SZ) running=${RUNNING[*]:-none} queue=${QUEUE[*]:-none}"
    LAST_PROGRESS=${NOW}
  fi
  sleep 60
done

echo "=== merging ==="
cd "${ROOT}"
"${PY}" -u -m validation.round8c --merge
echo "=== instances after round 8c ==="
gcloud compute instances list --project="${PROJECT}" || true
echo "=== round8c supervisor done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
