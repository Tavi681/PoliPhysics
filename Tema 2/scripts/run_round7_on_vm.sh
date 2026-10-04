#!/usr/bin/env bash
# In-VM Round-7 job sequence. Invoked under `timeout 86400` by the launcher.
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
LOG="${ROOT}/round7_export.log"

echo "=== round7 on-vm start $(date -Is) commit=${GIT_COMMIT} dirty=${CODE_DIRTY} jobs=${JOBS} ===" | tee -a "${LOG}"

python -u -m validation.round7_ns timing 2>&1 | tee -a "${LOG}"

python -u -m validation.export_paper --full --allow-dirty --jobs "${JOBS}" \
  --only tab_smith.csv,tab_smith_conv.csv,fig_smith.csv,fig_phase_maps/ \
  2>&1 | tee -a "${LOG}"

python -u -m validation.export_paper --full --allow-dirty --jobs "${JOBS}" \
  --only runs_round7/ \
  2>&1 | tee -a "${LOG}"

python -u -m validation.export_paper --full --allow-dirty --jobs "${JOBS}" \
  --only tab_mmin.csv \
  2>&1 | tee -a "${LOG}"

python -u -m validation.round7_ns check31 2>&1 | tee -a "${LOG}"
python -u -m validation.round7_ns maybe32 2>&1 | tee -a "${LOG}"

touch "${ROOT}/_ROUND7_DONE"
echo "=== round7 on-vm done $(date -Is) ===" | tee -a "${LOG}"
