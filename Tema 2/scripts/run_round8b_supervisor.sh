#!/usr/bin/env bash
# Mac-side Round-8b supervisor: quota → up to MAX_PAR parallel shards → pull+delete each.
# No wall-time cap. No pilot. Always delete the VM that this script created.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/.." && pwd)"
ZONE="${ZONE:-us-central1-a}"
PROJECT="${PROJECT:-authorship-verification}"
MACHINE_VCPU=16
LAUNCH="${ROOT}/scripts/run_round8b_export_on_gcp.sh"
LOG="${ROOT}/paper_results/round8b_supervisor.log"
export COPYFILE_DISABLE=1
mkdir -p "${ROOT}/paper_results"
exec > >(tee -a "${LOG}") 2>&1

echo "=== round8b supervisor start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
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
# MAX_PAR when our own r8b VMs are already counted in usage.
MAX_PAR=$(python3 -c "print(max(1, min(4, int(float(${C2_LIMIT}))//${MACHINE_VCPU})))")
echo "C2 limit=${C2_LIMIT} usage=${C2_USE} free=${FREE} max parallel=${MAX_PAR}"
if (( MAX_PAR < 4 )); then
  echo "Quota does not allow all 4 shards at once; queueing (cap ${MAX_PAR})."
fi

inst_name() {
  printf 'poliphysics-tema2-r8b-%s' "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
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
  gcloud compute ssh "${inst}" --zone="${ZONE}" --command="
      test -f /home/octav/tema2/_ROUND8B_${shard}_DONE
    " 2>/dev/null
}

pull_delete() {
  local shard="$1"
  local inst
  inst="$(inst_name "${shard}")"
  echo "Pull + delete ${inst}..."
  mkdir -p "${ROOT}/paper_results/round8b/${shard}"
  gcloud compute ssh "${inst}" --zone="${ZONE}" --command="
      tar -C /home/octav/tema2 -czf /tmp/tema2_r8b.tgz \
        paper_results/round8b/${shard} round8b_${shard}.log .cache/mmin
    " || true
  gcloud compute scp "${inst}:/tmp/tema2_r8b.tgz" \
    "/tmp/tema2_r8b_${shard}.tgz" --zone="${ZONE}" || true
  python3 - <<PY
from pathlib import Path
import tarfile
p = Path("/tmp/tema2_r8b_${shard}.tgz")
root = Path(r"${ROOT}")
if p.is_file():
    with tarfile.open(p) as t:
        t.extractall(root)
        print("extracted", p.name, "members", len(t.getnames()))
    njson = len(list((root/"paper_results/round8b"/"${shard}"/"runs").glob("*.json"))) if (root/"paper_results/round8b"/"${shard}"/"runs").exists() else 0
    print("row json files", njson)
PY
  if [[ -f "${ROOT}/round8b_${shard}.log" ]]; then
    cp "${ROOT}/round8b_${shard}.log" "${ROOT}/paper_results/round8b_${shard}_gcp.log"
  fi
  echo "Deleting ${inst} now."
  gcloud compute instances delete "${inst}" --zone="${ZONE}" --quiet
  gcloud compute instances list --project="${PROJECT}" || true
}

JOBS="${JOBS:-16}"
echo "Using JOBS=${JOBS}"

QUEUE=(B A D C)
declare -a RUNNING=()

still_needed() {
  local s="$1"
  [[ ! -f "${ROOT}/paper_results/round8b/${s}/.pulled" ]]
}

mark_pulled() {
  mkdir -p "${ROOT}/paper_results/round8b/$1"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${ROOT}/paper_results/round8b/$1/.pulled"
}

while true; do
  NEW_RUN=()
  for s in "${RUNNING[@]+"${RUNNING[@]}"}"; do
    if shard_done "${s}"; then
      pull_delete "${s}"
      mark_pulled "${s}"
    else
      echo "[sup] ${s} running $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      gcloud compute ssh "$(inst_name "${s}")" --zone="${ZONE}" \
        --command="tail -n 3 /home/octav/tema2/round8b_${s}.log" 2>/dev/null || true
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
  sleep 60
done

echo "=== merging ==="
cd "${ROOT}"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="${ROOT}/.venv/bin/python3"
fi
"${PY}" -u -m validation.round8b --merge
echo "=== instances after round 8b ==="
gcloud compute instances list --project="${PROJECT}" || true
echo "=== round8b supervisor done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
