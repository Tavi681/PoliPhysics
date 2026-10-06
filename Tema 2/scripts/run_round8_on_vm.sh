#!/usr/bin/env bash
# In-VM Round-8 shard. No wall-time cap.
#   SHARD=pilot|A|B|C|D bash scripts/run_round8_on_vm.sh
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
SHARD="${SHARD:-pilot}"
JOBS="${JOBS:-16}"
export JOBS
export ROUND8_OUT="${ROOT}/paper_results/round8"
LOG="${ROOT}/round8_${SHARD}.log"

echo "=== round8 on-vm start $(date -Is) shard=${SHARD} commit=${GIT_COMMIT} dirty=${CODE_DIRTY} jobs=${JOBS} ===" | tee -a "${LOG}"
python3 - <<'PY' | tee -a "${LOG}"
import os, json
print("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS"))
try:
    import threadpoolctl
    print("threadpool", json.dumps(threadpoolctl.threadpool_info(), default=str))
except Exception as e:
    print("threadpoolctl", e)
    import numpy as np
    np.show_config()
PY

python -u -m validation.round8 --shard "${SHARD}" 2>&1 | tee -a "${LOG}"

touch "${ROOT}/_ROUND8_${SHARD}_DONE"
echo "=== round8 on-vm done $(date -Is) shard=${SHARD} ===" | tee -a "${LOG}"
