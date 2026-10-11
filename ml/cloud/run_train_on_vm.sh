#!/usr/bin/env bash
# Runs on the GPU VM: sync data, compute noise_stats, train configs one after another.
# Env: DATA_VERSION, BUCKET, CONFIGS, UTIL_GATE_MIN, UTIL_GATE_PCT, REMOTE_DIR
set -euo pipefail
cd "${REMOTE_DIR:-/home/octav/poliphysics}"
export PYTHONPATH=.
export WANDB_PROJECT="${WANDB_PROJECT:-poliphysics-ballooning-ml}"

DATA_VERSION="${DATA_VERSION:-5c60fa669a719404}"
BUCKET="${BUCKET:-gs://authorship-verification-poliphysics/tema1-ml}"
DEST="/data/processed/${DATA_VERSION}"
CONFIGS="${CONFIGS:-ml/configs/gns_local_gpu.yaml ml/configs/gns_baseline_gpu.yaml}"
UTIL_GATE_MIN="${UTIL_GATE_MIN:-15}"
UTIL_GATE_PCT="${UTIL_GATE_PCT:-70}"
HOUR_REPORT_S="${HOUR_REPORT_S:-3600}"

mkdir -p "${DEST}" "Tema 1/results/ml/runs" "Tema 1/results/ml/logs"
if [[ ! -f "${DEST}/DATA_VERSION" ]]; then
  echo "Syncing ${BUCKET}/processed/${DATA_VERSION} → ${DEST}"
  gsutil -m rsync -r "${BUCKET}/processed/${DATA_VERSION}" "${DEST}"
fi

python3 - <<'PY'
from pathlib import Path
from ml.data.dataset import BallooningRunStore
from ml.data.noise_stats import compute_dv_std_by_type, save_noise_stats
from ml.data.splits import load_splits
root = Path("/data/processed") / __import__("os").environ.get("DATA_VERSION", "5c60fa669a719404")
out = root / "noise_stats.json"
if out.exists():
    print("noise_stats exists", out)
else:
    store = BallooningRunStore(root, cache_shards=2)
    splits = load_splits(root / "splits" / "splits.json")
    # subsample train for wall-clock; 400 runs × 50 frames is enough for stds
    train = list(splits["train"])
    step = max(len(train) // 400, 1)
    sample = train[::step][:400]
    stats = compute_dv_std_by_type(store, sample, max_frames_per_run=50)
    save_noise_stats(stats, out)
    print(stats)
PY

python3 -m pip install -q wandb pyyaml 2>/dev/null || true

first=1
for CFG in ${CONFIGS}; do
  echo "=== TRAIN ${CFG} @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  LOG="Tema 1/results/ml/logs/$(basename "${CFG}" .yaml)_train.log"
  python3 -m ml.train --config "${CFG}" --device cuda 2>&1 | tee "${LOG}" &
  TPID=$!
  # sample GPU util for the first UTIL_GATE_MIN minutes (local ablation first)
  if [[ ${first} -eq 1 ]]; then
    first=0
    end=$((SECONDS + UTIL_GATE_MIN * 60))
    sum=0; n=0
    while (( SECONDS < end )); do
      if ! kill -0 "${TPID}" 2>/dev/null; then break; fi
      u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
      if [[ -n "${u}" ]]; then sum=$((sum + u)); n=$((n + 1)); fi
      sleep 30
    done
    avg=0
    (( n > 0 )) && avg=$((sum / n))
    echo "UTIL_GATE_AVG_${UTIL_GATE_MIN}m=${avg} (n=${n})"
    if (( n > 0 && avg < UTIL_GATE_PCT )); then
      echo "GPU util ${avg}% < ${UTIL_GATE_PCT}% — STOP. Propose g2-standard-16; do not auto-switch."
      kill "${TPID}" 2>/dev/null || true
      wait "${TPID}" 2>/dev/null || true
      exit 42
    fi
  fi
  # 1h progress snapshot (non-fatal)
  (
    sleep "${HOUR_REPORT_S}"
    echo "=== 1h REPORT $(basename "${CFG}") $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv || true
    tail -n 40 "${LOG}" || true
  ) &
  wait "${TPID}"
  RUN=$(ls -1dt "Tema 1/results/ml/runs"/*/ | head -1)
  echo "Evaluating ${RUN}"
  python3 -m ml.evaluate --config "${CFG}" --checkpoint "${RUN}/best.pt" --device cuda \
    --splits test_interp test_unseenN_interp test_unseenN test_unseenSigma
  RID=$(basename "${RUN}")
  gsutil -m rsync -r "${RUN}" "${BUCKET}/runs/${RID}/"
  gsutil -m rsync -r "Tema 1/results/ml/${RID}" "${BUCKET}/runs/${RID}/metrics/" 2>/dev/null || true
done
echo DONE_TRAIN
