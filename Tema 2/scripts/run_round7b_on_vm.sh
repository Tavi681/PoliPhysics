#!/usr/bin/env bash
# In-VM Round-7b job: estimate → tab_mmin n_s=10 → 3.1 → reduced 3.2.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/netsim_mpl
export GIT_COMMIT="${GIT_COMMIT:-$(tr -d '[:space:]' < COMMIT 2>/dev/null || echo unknown)}"
export CODE_DIRTY="${CODE_DIRTY:-$(tr -d '[:space:]' < CODE_DIRTY 2>/dev/null || echo true)}"
JOBS="${JOBS:-16}"
LOG="${ROOT}/round7b_export.log"

echo "=== round7b on-vm start $(date -Is) commit=${GIT_COMMIT} dirty=${CODE_DIRTY} jobs=${JOBS} ===" | tee -a "${LOG}"

python -u -m validation.round7_ns estimate 2>&1 | tee -a "${LOG}"

export MMIN_N_PROCS=1
echo "=== 7b (a) tab_mmin n_s=10, 36 rows, tol=0.01 ===" | tee -a "${LOG}"
python -u -m validation.export_paper --full --allow-dirty --jobs "${JOBS}" \
  --only tab_mmin.csv 2>&1 | tee -a "${LOG}"

# Inner Alg.2 parallelism (3 impact points + B-scan) on a single parent process.
export MMIN_N_PROCS="${JOBS}"
python -u -m validation.round7_ns check31 2>&1 | tee -a "${LOG}"

# Independent 3.2 jobs in an outer pool; inner n_procs=1 (daemon workers).
export MMIN_N_PROCS=1
python -u -m validation.round7_ns maybe32 2>&1 | tee -a "${LOG}"

touch "${ROOT}/_ROUND7B_DONE"
echo "=== round7b on-vm done $(date -Is) ===" | tee -a "${LOG}"
