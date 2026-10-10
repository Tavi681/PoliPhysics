#!/usr/bin/env bash
# Runs on the preprocess VM (installed by run_preprocess_on_gcp.sh).
set -euo pipefail

ROOT="${ROOT:-/home/octav/poliphysics}"
OUT="${OUT:-/data/processed}"
WORKERS="${WORKERS:-16}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CLOUDSDK_CORE_DISABLE_FILE_LOGGING=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${ROOT}"
START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s)"
GIT_COMMIT="$(cat "${ROOT}/COMMIT" 2>/dev/null || git rev-parse HEAD)"
CODE_DIRTY="$(cat "${ROOT}/CODE_DIRTY" 2>/dev/null || echo unknown)"

mkdir -p "${OUT}" /data/logs
LOG=/data/logs/preprocess.log
exec > >(tee -a "${LOG}") 2>&1

echo "=== preprocess start ${START_TS} commit=${GIT_COMMIT} dirty=${CODE_DIRTY} ==="
python3 -m venv /data/venv
source /data/venv/bin/activate
pip install -U pip wheel
# CPU torch wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install h5py numpy pandas tqdm gcsfs google-cloud-storage PyYAML scipy

echo "=== build splits ==="
python -m ml.scripts.build_splits \
  --config ml/configs/splits_default.yaml \
  --out "${OUT}/splits"

echo "=== preprocess all valid runs (stream GCS, resume) ==="
python -m ml.data.preprocess \
  --out "${OUT}" \
  --splits "${OUT}/splits/splits.json" \
  --workers "${WORKERS}" \
  --shard-size 50 \
  --upload-gcs

DATA_VERSION="$(cat "${OUT}/DATA_VERSION" | tr -d '\n')"
echo "DATA_VERSION=${DATA_VERSION}"

echo "=== verify against raw HDF5 ==="
python -m ml.cloud.verify_processed \
  --data-version "${DATA_VERSION}" \
  --out "${OUT}"

END_EPOCH="$(date +%s)"
WALL=$((END_EPOCH - START_EPOCH))
# c2-standard-16 us-central1 ≈ $0.6688/h; 100GB pd-balanced ≈ $0.10/GB-mo
COST_VM=$(python3 -c "print(round(${WALL}/3600.0 * 0.6688, 4))")
COST_DISK=$(python3 -c "print(round(${WALL}/3600.0 * (100*0.10)/(24*30), 4))")
python3 - <<PY
import json
from pathlib import Path
rep = {
    "git_commit": "${GIT_COMMIT}",
    "code_dirty": "${CODE_DIRTY}" == "true",
    "start_utc": "${START_TS}",
    "wall_time_s": ${WALL},
    "data_version": "${DATA_VERSION}",
    "machine": "c2-standard-16",
    "zone": "us-central1-a",
    "disk_gb": 100,
    "mode": "stream_gcs",
    "cost_usd_est": {"vm": ${COST_VM}, "disk": ${COST_DISK}, "total": round(${COST_VM}+${COST_DISK}, 4)},
}
Path("${OUT}/run_report.json").write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
# also upload report next to processed
from google.cloud import storage
b = storage.Client().bucket("authorship-verification-poliphysics")
b.blob(f"tema1-ml/processed/${DATA_VERSION}/run_report.json").upload_from_filename("${OUT}/run_report.json")
b.blob(f"tema1-ml/processed/${DATA_VERSION}/verify_report.json").upload_from_filename("${OUT}/verify_report.json")
print("DONE")
PY
