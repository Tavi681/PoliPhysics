#!/usr/bin/env bash
# In-VM Round-8c shard (post-arrest + slow timeouts). No wall-time cap.
#   SHARD=A|B|C|D bash scripts/run_round8c_on_vm.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/netsim_mpl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export GIT_COMMIT="${GIT_COMMIT:-$(tr -d '[:space:]' < COMMIT 2>/dev/null || echo unknown)}"
export CODE_DIRTY="${CODE_DIRTY:-$(tr -d '[:space:]' < CODE_DIRTY 2>/dev/null || echo true)}"
SHARD="${SHARD:-A}"
JOBS="${JOBS:-16}"
export JOBS
export ROUND8C_OUT="${ROOT}/paper_results/round8c"
LOG="${ROOT}/round8c_${SHARD}.log"

echo "=== round8c on-vm start $(date -u +%Y-%m-%dT%H:%M:%SZ) shard=${SHARD} commit=${GIT_COMMIT} dirty=${CODE_DIRTY} jobs=${JOBS} ===" | tee -a "${LOG}"
python -u -m validation.round8c --plan 2>&1 | tee -a "${LOG}"
python -u -m validation.round8c --shard "${SHARD}" 2>&1 | tee -a "${LOG}"

touch "${ROOT}/_ROUND8C_${SHARD}_DONE"
echo "=== round8c on-vm done $(date -u +%Y-%m-%dT%H:%M:%SZ) shard=${SHARD} ===" | tee -a "${LOG}"
