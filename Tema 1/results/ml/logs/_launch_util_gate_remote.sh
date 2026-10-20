#!/usr/bin/env bash
set -euo pipefail
cd /home/octav/poliphysics
mkdir -p "Tema 1/results/ml/logs"
pgrep -af '[m]l\.train' && pkill -f '[m]l\.train' || true
sleep 2
set -a
source ~/.config/wandb_env
set +a
python3 -m pip install -q nvidia-ml-py psutil wandb 2>/dev/null || true
nohup env PYTHONPATH=. PYTHONUNBUFFERED=1 \
  UTIL_GATE=1 UTIL_GATE_MIN=15 UTIL_GATE_PCT=70 UTIL_GATE_VAL_FRAC=0.10 \
  WANDB_MODE=online WANDB_PROJECT=poliphysics-ballooning-ml \
  WANDB_API_KEY="${WANDB_API_KEY}" \
  python3 -m ml.train --config ml/configs/gns_local_gpu.yaml --device cuda \
  > "Tema 1/results/ml/logs/util_gate_v2.log" 2>&1 &
echo "GATE_PID=$!"
echo "GATE_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
sleep 5
pgrep -af '[m]l\.train' || true
tail -n 25 "Tema 1/results/ml/logs/util_gate_v2.log" || true
