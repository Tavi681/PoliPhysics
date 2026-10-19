#!/usr/bin/env bash
# After gns_baseline finishes: long-horizon ckpt selection → full paper eval → GCS → stop VM.
# Sequential on the GPU; never parallel with training.
set -euo pipefail
cd "${REMOTE_DIR:-/home/octav/poliphysics}/Tema 1"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
BUCKET="${BUCKET:-gs://authorship-verification-poliphysics/tema1-ml}"
LOCAL_RID="gns_local_s42_88dc163f_1791760366"
BASE_RID="${BASE_RID:-gns_baseline_s42_5716ee5b_1791828699}"
ZONE="${ZONE:-us-central1-b}"
INSTANCE="${INSTANCE:-poliphysics-tema1-ml-train}"

LOCAL_RUN="results/ml/runs/${LOCAL_RID}"
BASE_RUN="results/ml/runs/${BASE_RID}"
mkdir -p "results/ml/logs" "results/ml/runs"

echo "=== post pipeline LOCAL=${LOCAL_RID} BASE=${BASE_RID} @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# --- 3. Long-horizon checkpoint selection ---
LOCAL_CKPTS=()
[[ -f "${LOCAL_RUN}/best.pt" ]] && LOCAL_CKPTS+=("${LOCAL_RUN}/best.pt")
[[ -f "${LOCAL_RUN}/last.pt" ]] && LOCAL_CKPTS+=("${LOCAL_RUN}/last.pt")
BASE_CKPTS=()
shopt -s nullglob
for f in "${BASE_RUN}"/epoch_*.pt; do BASE_CKPTS+=("$f"); done
[[ -f "${BASE_RUN}/best.pt" ]] && BASE_CKPTS+=("${BASE_RUN}/best.pt")
[[ -f "${BASE_RUN}/last.pt" ]] && BASE_CKPTS+=("${BASE_RUN}/last.pt")
shopt -u nullglob

echo "local ckpts: ${LOCAL_CKPTS[*]}"
echo "baseline ckpts: ${BASE_CKPTS[*]}"

python3 -m ml.scripts.select_checkpoint_horizon \
  --config ml/configs/gns_local_gpu.yaml \
  --checkpoints "${LOCAL_CKPTS[@]}" \
  --baselines --device cuda --t-end 60 \
  --out "results/ml/logs/ckpt_select_${LOCAL_RID}.json"

python3 -m ml.scripts.select_checkpoint_horizon \
  --config ml/configs/gns_baseline_gpu.yaml \
  --checkpoints "${BASE_CKPTS[@]}" \
  --baselines --device cuda --t-end 60 \
  --out "results/ml/logs/ckpt_select_${BASE_RID}.json"

LOCAL_SEL=$(python3 -c "import json; d=json.load(open('results/ml/logs/ckpt_select_${LOCAL_RID}.json')); print(d['checkpoints'][d['selected']]['path'])")
BASE_SEL=$(python3 -c "import json; d=json.load(open('results/ml/logs/ckpt_select_${BASE_RID}.json')); print(d['checkpoints'][d['selected']]['path'])")
echo "SELECTED local=${LOCAL_SEL}"
echo "SELECTED baseline=${BASE_SEL}"

make_baseline_ckpt() {
  local NAME="$1" CFG="$2" OUTDIR="$3"
  python3 - <<PY
from pathlib import Path
import torch
from ml.utils.config import load_yaml
from ml.train import _resolve
from ml.data.dataset import BallooningRunStore
from ml.models import build_model
cfg = load_yaml(_resolve("${CFG}"))
dd = cfg["data"]["data_dir"]
store = BallooningRunStore(dd if Path(dd).is_absolute() else _resolve(dd))
m = build_model("${NAME}", a_mean=store.norm["a"]["mean"], a_std=store.norm["a"]["std"])
out = Path("${OUTDIR}")
out.mkdir(parents=True, exist_ok=True)
torch.save({"model": m.state_dict(), "meta": {"run_id": out.name, "train_time_s": 0.0}, "cfg": cfg}, out / "best.pt")
print(out / "best.pt")
PY
}

sync_eval() {
  local CKPT="$1" EXTRA_JSON="${2:-}"
  local RID
  RID=$(python3 -c "import torch; b=torch.load('${CKPT}', map_location='cpu', weights_only=False); print((b.get('meta') or {}).get('run_id') or '')")
  [[ -z "${RID}" ]] && RID=$(basename "$(dirname "${CKPT}")")
  local EVAL_DIR="results/ml/${RID}"
  local RUN_DIR="results/ml/runs/${RID}"
  mkdir -p "${RUN_DIR}"
  if [[ -d "${EVAL_DIR}" ]]; then
    cp -f "${EVAL_DIR}/metrics.json" "${RUN_DIR}/" 2>/dev/null || true
    cp -f "${EVAL_DIR}/tab_accuracy.csv" "${RUN_DIR}/" 2>/dev/null || true
    cp -f "${EVAL_DIR}/tab_cost.csv" "${RUN_DIR}/" 2>/dev/null || true
    gsutil -m rsync -r "${EVAL_DIR}" "${BUCKET}/runs/${RID}/eval/" || true
  fi
  gsutil -m rsync -r "${RUN_DIR}" "${BUCKET}/runs/${RID}/" || true
  if [[ -n "${EXTRA_JSON}" && -f "${EXTRA_JSON}" ]]; then
    gsutil cp "${EXTRA_JSON}" "${BUCKET}/runs/${RID}/" || true
  fi
}

eval_one() {
  local CFG="$1" CKPT="$2" TAG="$3"
  echo "=== FULL EVAL ${TAG} @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ckpt=${CKPT} ==="
  local T0=$(date +%s)
  python3 -m ml.evaluate --config "${CFG}" --checkpoint "${CKPT}" --device cuda \
    --splits test_interp test_unseenN_interp test_unseenN test_unseenSigma
  local T1=$(date +%s)
  echo "EVAL_WALL_S_${TAG}=$((T1 - T0))"
  echo "EVAL_COST_USD_${TAG}=$(python3 -c "print(round($((T1-T0))/3600*0.853624312, 3))")"
}

# --- 4. Full paper evaluation ---
eval_one ml/configs/gns_local_gpu.yaml "${LOCAL_SEL}" gns_local
sync_eval "${LOCAL_SEL}" "results/ml/logs/ckpt_select_${LOCAL_RID}.json"

eval_one ml/configs/gns_baseline_gpu.yaml "${BASE_SEL}" gns_baseline
sync_eval "${BASE_SEL}" "results/ml/logs/ckpt_select_${BASE_RID}.json"

CV_CKPT=$(make_baseline_ckpt constant_velocity ml/configs/baseline_const_gpu.yaml \
  "results/ml/runs/_baseline_const_paper")
FA_CKPT=$(make_baseline_ckpt follow_air ml/configs/baseline_follow_air_gpu.yaml \
  "results/ml/runs/_baseline_follow_air_paper")
eval_one ml/configs/baseline_const_gpu.yaml "${CV_CKPT}" constant_velocity
sync_eval "${CV_CKPT}"
eval_one ml/configs/baseline_follow_air_gpu.yaml "${FA_CKPT}" follow_air
sync_eval "${FA_CKPT}"

# Sync epoch checkpoints if any
gsutil -m rsync -r "${BASE_RUN}" "${BUCKET}/runs/${BASE_RID}/" || true
gsutil -m rsync -r "${LOCAL_RUN}" "${BUCKET}/runs/${LOCAL_RID}/" || true

echo DONE_POST_PIPELINE
gcloud compute instances stop "${INSTANCE}" --zone="${ZONE}" || true
echo VM_STOPPED
