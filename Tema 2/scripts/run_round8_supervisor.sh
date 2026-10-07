#!/usr/bin/env bash
# Mac-side Round-8 supervisor: quota → pilot → up to 2 parallel shards → pull+delete each.
# C2_CPUS remaining after Tema 1 (2×c2-standard-30) is 40, so max 2 × c2-standard-16.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/.." && pwd)"
ZONE="${ZONE:-us-central1-a}"
PROJECT="${PROJECT:-authorship-verification}"
MACHINE_VCPU=16
LAUNCH="${ROOT}/scripts/run_round8_export_on_gcp.sh"
LOG="${ROOT}/paper_results/round8_supervisor.log"
mkdir -p "${ROOT}/paper_results"
exec > >(tee -a "${LOG}") 2>&1

echo "=== round8 supervisor start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
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
# Tema 1 currently holds 60 C2; leftover 40 ⇒ two c2-standard-16. Do not
# shrink the cap when our own r8 VMs are already counted in usage.
FREE_AFTER_TEMA1=40
MAX_PAR=$(python3 -c "print(max(1, min(4, int(${FREE_AFTER_TEMA1})//${MACHINE_VCPU})))")
echo "C2 limit=${C2_LIMIT} usage=${C2_USE} planned max parallel=${MAX_PAR} (40 leftover after Tema 1)"
if (( MAX_PAR < 4 )); then
  echo "Quota does not allow all 4 shards at once; queueing (cap ${MAX_PAR})."
fi

inst_name() {
  printf 'poliphysics-tema2-r8-%s' "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
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
      test -f /home/octav/tema2/_ROUND8_${shard}_DONE
    " 2>/dev/null
}

pull_delete() {
  local shard="$1"
  local inst
  inst="$(inst_name "${shard}")"
  echo "Pull + delete ${inst}..."
  DETACH=false SHARD="${shard}" bash -c '
    source /dev/null
  ' || true
  # Pull via the launcher poll path: ssh tar + scp + extract, then delete.
  mkdir -p "${ROOT}/paper_results/round8/${shard}"
  gcloud compute ssh "${inst}" --zone="${ZONE}" --command="
      tar -C /home/octav/tema2 -czf /tmp/tema2_r8.tgz \
        paper_results/round8/${shard} round8_${shard}.log \
        paper_results/params_used.csv paper_results/tab_conv.csv \
        paper_results/fig_etaa.csv 2>/dev/null || \
      tar -C /home/octav/tema2 -czf /tmp/tema2_r8.tgz \
        paper_results/round8/${shard} round8_${shard}.log
    " || true
  gcloud compute scp "${inst}:/tmp/tema2_r8.tgz" \
    "/tmp/tema2_r8_${shard}.tgz" --zone="${ZONE}" || true
  python3 - <<PY
from pathlib import Path
import tarfile
p = Path("/tmp/tema2_r8_${shard}.tgz")
root = Path(r"${ROOT}")
if p.is_file():
    with tarfile.open(p) as t:
        t.extractall(root)
        print("extracted", p.name, "members", len(t.getnames()))
    njson = len(list((root/"paper_results/round8"/"${shard}"/"runs").glob("*.json"))) if (root/"paper_results/round8"/"${shard}"/"runs").exists() else 0
    print("row json files", njson)
PY
  if [[ -f "${ROOT}/round8_${shard}.log" ]]; then
    cp "${ROOT}/round8_${shard}.log" "${ROOT}/paper_results/round8_${shard}_gcp.log"
  fi
  echo "Deleting ${inst} now."
  gcloud compute instances delete "${inst}" --zone="${ZONE}" --quiet
  gcloud compute instances list --project="${PROJECT}" || true
}

# --- pilot ---
if [[ ! -f "${ROOT}/paper_results/round8/pilot/pilot.json" ]]; then
  launch_shard pilot 16
  echo "Waiting for pilot..."
  while ! shard_done pilot; do
    echo "[sup] pilot still running $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    gcloud compute ssh "$(inst_name pilot)" --zone="${ZONE}" \
      --command="tail -n 4 /home/octav/tema2/round8_pilot.log" 2>/dev/null || true
    sleep 45
  done
  pull_delete pilot
fi
JOBS=16
if [[ -f "${ROOT}/paper_results/round8/pilot/pilot.json" ]]; then
  JOBS=$(python3 -c "import json; print(json.load(open('${ROOT}/paper_results/round8/pilot/pilot.json')).get('better_jobs',16))")
fi
echo "Using JOBS=${JOBS} for remaining shards"

QUEUE=(D B C A)
declare -a RUNNING=()

still_needed() {
  local s="$1"
  [[ ! -f "${ROOT}/paper_results/round8/${s}/.pulled" ]]
}

mark_pulled() {
  mkdir -p "${ROOT}/paper_results/round8/$1"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${ROOT}/paper_results/round8/$1/.pulled"
}

while true; do
  # Reap finished
  NEW_RUN=()
  for s in "${RUNNING[@]+"${RUNNING[@]}"}"; do
    if shard_done "${s}"; then
      pull_delete "${s}"
      mark_pulled "${s}"
    else
    echo "[sup] ${s} running $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      gcloud compute ssh "$(inst_name "${s}")" --zone="${ZONE}" \
        --command="tail -n 3 /home/octav/tema2/round8_${s}.log" 2>/dev/null || true
      NEW_RUN+=("${s}")
    fi
  done
  RUNNING=("${NEW_RUN[@]+"${NEW_RUN[@]}"}")

  # Launch while under cap
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
"${PY}" -u -m validation.round8 --merge
echo "=== instances after round 8 ==="
gcloud compute instances list --project="${PROJECT}" || true
echo "=== round8 supervisor done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
