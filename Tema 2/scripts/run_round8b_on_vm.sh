#!/usr/bin/env bash
# In-VM Round-8b shard (Alg.2 a/R extension). No wall-time cap.
#   SHARD=A|B|C|D bash scripts/run_round8b_on_vm.sh
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
export MMIN_N_PROCS="${MMIN_N_PROCS:-8}"
export ROUND8B_OUT="${ROOT}/paper_results/round8b"
export MMIN_CACHE="${ROOT}/.cache/mmin"
LOG="${ROOT}/round8b_${SHARD}.log"

echo "=== round8b on-vm start $(date -u +%Y-%m-%dT%H:%M:%SZ) shard=${SHARD} commit=${GIT_COMMIT} dirty=${CODE_DIRTY} jobs=${JOBS} mmin_n_procs=${MMIN_N_PROCS} ===" | tee -a "${LOG}"
python3 - <<'PY' | tee -a "${LOG}"
import os
from pathlib import Path
print("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS"))
print("MMIN_N_PROCS", os.environ.get("MMIN_N_PROCS"))
c = Path(".cache/mmin")
n = sum(1 for _ in c.rglob("mmin_p*.h5")) if c.is_dir() else 0
print("mmin cache h5", n)
PY

python -u -m validation.round8b --shard "${SHARD}" 2>&1 | tee -a "${LOG}"

touch "${ROOT}/_ROUND8B_${SHARD}_DONE"
echo "=== round8b on-vm done $(date -u +%Y-%m-%dT%H:%M:%SZ) shard=${SHARD} ===" | tee -a "${LOG}"
