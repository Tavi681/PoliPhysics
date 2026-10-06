#!/usr/bin/env bash
# In-VM Round-7c job: print cache count → estimate → S n_s check → table.
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
export JOBS
LOG="${ROOT}/round7c_export.log"

echo "=== round7c on-vm start $(date -Is) commit=${GIT_COMMIT} dirty=${CODE_DIRTY} jobs=${JOBS} ===" | tee -a "${LOG}"
N_H5="$(find "${ROOT}/.cache/mmin" -name 'mmin_p*.h5' 2>/dev/null | wc -l | tr -d ' ')"
echo "=== cached mmin .h5 files on VM: ${N_H5} (path ${ROOT}/.cache/mmin) ===" | tee -a "${LOG}"

python -u -m validation.round7c round7c 2>&1 | tee -a "${LOG}"

touch "${ROOT}/_ROUND7C_DONE"
echo "=== round7c on-vm done $(date -Is) ===" | tee -a "${LOG}"
