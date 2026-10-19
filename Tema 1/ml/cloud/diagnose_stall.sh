#!/usr/bin/env bash
# Stall diagnosis on the GPU VM: phase timestamps + RAM sampling + py-spy dumps.
# Usage (on VM): bash ml/cloud/diagnose_stall.sh [config]
set -euo pipefail
cd "${REMOTE_DIR:-/home/octav/poliphysics}/Tema 1"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export MEM_SAMPLE_EVERY_S=10
CFG="${1:-ml/configs/gns_local_gpu.yaml}"
OUT="results/ml/logs/stall_diag_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${OUT}"
LOG="${OUT}/train.log"
echo "OUT=${OUT} CFG=${CFG}"

# Estimate per-shard mmap size (file size) for cache budget report
python3 - <<'PY' "${CFG}" "${OUT}/cache_budget.json"
import json, sys
from pathlib import Path
from ml.utils.config import load_yaml
from ml.utils.git_info import repo_root
cfg_path = Path(sys.argv[1])
out = Path(sys.argv[2])
cfg = load_yaml(cfg_path if cfg_path.is_absolute() else repo_root() / cfg_path)
dcfg, tcfg = cfg["data"], cfg["train"]
root = Path(dcfg["data_dir"])
man = json.loads((root / "manifest.json").read_text())
sizes = []
for sh in man.get("shards", [])[:32]:
    p = root / sh
    if p.exists():
        sizes.append(p.stat().st_size)
avg = sum(sizes) / max(len(sizes), 1)
cache = int(dcfg.get("cache_shards", 4))
workers = int(tcfg.get("num_workers", 0))
# With mmap, physical RAM is shared via page cache; worst-case private expand would be:
report = {
    "avg_shard_bytes": avg,
    "cache_shards": cache,
    "num_workers": workers,
    "worst_case_private_gb": avg * cache * max(workers, 1) / 1e9,
    "mmap_note": "mmap shares OS page cache across workers; private expand would multiply",
}
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
PY

python3 -m pip install -q py-spy psutil nvidia-ml-py 2>/dev/null || true

# Start training in background
python3 -m ml.train --config "${CFG}" --device cuda > "${LOG}" 2>&1 &
TPID=$!
echo "TPID=${TPID}"
echo "${TPID}" > "${OUT}/train.pid"

# py-spy dumps at 2 min and 5 min for main + workers
(
  sleep 120
  if kill -0 "${TPID}" 2>/dev/null; then
    echo "py-spy @ 2min" | tee -a "${OUT}/pyspy_times.txt"
    py-spy dump --pid "${TPID}" > "${OUT}/pyspy_main_2m.txt" 2>&1 || true
    for c in $(pgrep -P "${TPID}" 2>/dev/null || true); do
      py-spy dump --pid "${c}" > "${OUT}/pyspy_worker_${c}_2m.txt" 2>&1 || true
    done
  fi
  sleep 180
  if kill -0 "${TPID}" 2>/dev/null; then
    echo "py-spy @ 5min" | tee -a "${OUT}/pyspy_times.txt"
    py-spy dump --pid "${TPID}" > "${OUT}/pyspy_main_5m.txt" 2>&1 || true
    for c in $(pgrep -P "${TPID}" 2>/dev/null || true); do
      py-spy dump --pid "${c}" > "${OUT}/pyspy_worker_${c}_5m.txt" 2>&1 || true
    done
  fi
) &
SPY=$!

# Run for up to 8 minutes of diagnosis, then stop
end=$((SECONDS + 480))
while (( SECONDS < end )); do
  if ! kill -0 "${TPID}" 2>/dev/null; then break; fi
  # stop early once we have >= 20 steps
  if grep -qE '^STEP 20 ' "${LOG}" 2>/dev/null; then
    echo "reached step 20 — diagnosis sufficient"
    break
  fi
  sleep 5
done

kill "${TPID}" 2>/dev/null || true
wait "${TPID}" 2>/dev/null || true
kill "${SPY}" 2>/dev/null || true
wait "${SPY}" 2>/dev/null || true

echo "=== PHASE / STEP excerpt ==="
grep -E '^(PHASE|STEP|MEM|UTIL)' "${LOG}" | head -200 || true
echo "DONE_DIAG out=${OUT}"
