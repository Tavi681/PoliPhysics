#!/usr/bin/env bash
# On-VM: inspect shard overhead → re-pack+u_next → upload → verify → report.
set -euo pipefail

ROOT="${ROOT:-/home/octav/poliphysics}"
OUT="${OUT:-/data/repack}"
SRC_VERSION="${SRC_VERSION:-b0a7c5b503af5d26}"
WORKERS="${WORKERS:-8}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CLOUDSDK_CORE_DISABLE_FILE_LOGGING=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${ROOT}"
START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s)"
GIT_COMMIT="$(cat "${ROOT}/ml/COMMIT" 2>/dev/null || echo unknown)"
CODE_DIRTY="$(cat "${ROOT}/ml/CODE_DIRTY" 2>/dev/null || echo unknown)"

mkdir -p "${OUT}" /data/logs
LOG=/data/logs/repack.log
exec > >(tee -a "${LOG}") 2>&1

echo "=== repack start ${START_TS} commit=${GIT_COMMIT} dirty=${CODE_DIRTY} src=${SRC_VERSION} ==="
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip python3-dev build-essential libhdf5-dev pkg-config
if [[ ! -d /data/venv ]]; then
  python3 -m venv /data/venv
fi
source /data/venv/bin/activate
pip install -U pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install h5py numpy pandas tqdm gcsfs google-cloud-storage PyYAML scipy

echo "=== shard overhead inspect (3 shards) ==="
mkdir -p /data/shard_inspect
export SRC_VERSION
python3 - <<'PY'
import gcsfs, json, os
from pathlib import Path
from ml.data.shard_inspect import inspect_shard
from ml.data.preprocess import GCS_BUCKET, GCS_PROCESSED_PREFIX
src = os.environ.get("SRC_VERSION", "b0a7c5b503af5d26")
fs = gcsfs.GCSFileSystem()
prefix = f"{GCS_BUCKET}/{GCS_PROCESSED_PREFIX}/{src}"
# pick first, mid, last shard by name
names = sorted(
    p.rsplit("/", 1)[-1]
    for p in fs.ls(prefix)
    if p.rsplit("/", 1)[-1].startswith("shard_") and p.rsplit("/", 1)[-1].endswith(".pt")
)
pick = [names[0], names[len(names)//2], names[-1]]
out_dir = Path("/data/shard_inspect")
reports = []
for n in pick:
    local = out_dir / n
    if not local.exists():
        fs.get(f"{prefix}/{n}", str(local))
    rep = inspect_shard(local)
    reports.append(rep)
    print(json.dumps({k: rep[k] for k in (
        "path", "file_size", "n_runs", "sum_tensor_nbytes",
        "sum_storage_nbytes_raw", "sum_storage_nbytes_dedup",
        "overhead_file_vs_nbytes", "overhead_file_vs_storage_dedup",
        "n_array_objects", "n_views_or_slices", "kinds_present", "dtypes_present",
    )}, indent=2), flush=True)
(out_dir / "overhead_report.json").write_text(json.dumps(reports, indent=2))
print("WROTE", out_dir / "overhead_report.json", flush=True)
PY

echo "=== re-pack + u_next (resume) ==="
python -m ml.data.repack \
  --src-version "${SRC_VERSION}" \
  --out "${OUT}" \
  --workers "${WORKERS}" \
  --upload-gcs

DATA_VERSION="$(cat "${OUT}/DATA_VERSION" | tr -d '\n')"
echo "NEW_DATA_VERSION=${DATA_VERSION}"

echo "=== verify new vs HDF5 (3/N) ==="
python -m ml.cloud.verify_processed \
  --data-version "${DATA_VERSION}" \
  --out "${OUT}"

# Upload overhead report next to new version
python3 - <<PY
import json
from pathlib import Path
from google.cloud import storage
b = storage.Client().bucket("authorship-verification-poliphysics")
dv = Path("${OUT}/DATA_VERSION").read_text().strip()
for rel, local in [
    ("overhead_report.json", "/data/shard_inspect/overhead_report.json"),
    ("verify_report.json", "${OUT}/verify_report.json"),
]:
    p = Path(local)
    if p.exists():
        b.blob(f"tema1-ml/processed/{dv}/{rel}").upload_from_filename(str(p))
print("uploaded aux reports", flush=True)
PY

END_EPOCH="$(date +%s)"
WALL=$((END_EPOCH - START_EPOCH))
# c2-standard-16 ≈ \$0.6688/h; 200GB pd-balanced ≈ \$0.10/GB-mo
COST_VM=$(python3 -c "print(round(${WALL}/3600.0 * 0.6688, 4))")
COST_DISK=$(python3 -c "print(round(${WALL}/3600.0 * (200*0.10)/(24*30), 4))")
python3 - <<PY
import json
from pathlib import Path
from google.cloud import storage
man = json.loads(Path("${OUT}/manifest.json").read_text())
ov = {}
op = Path("/data/shard_inspect/overhead_report.json")
if op.exists():
    ov = {"shards": json.loads(op.read_text())}
rep = {
    "git_commit": "${GIT_COMMIT}",
    "code_dirty": "${CODE_DIRTY}" == "true",
    "start_utc": "${START_TS}",
    "wall_time_s": ${WALL},
    "src_version": "${SRC_VERSION}",
    "data_version": "$(cat ${OUT}/DATA_VERSION | tr -d '\n')",
    "machine": "c2-standard-16",
    "zone": "us-central1-a",
    "disk_gb": 200,
    "mode": "repack_u_next",
    "store_u_next": True,
    "leak_test": man.get("leak_test"),
    "repack_max_abs_vs_src": man.get("repack_max_abs_vs_src"),
    "overhead_inspect": [
        {k: s[k] for k in (
            "path", "file_size", "sum_tensor_nbytes", "sum_storage_nbytes_dedup",
            "n_views_or_slices", "dtypes_present", "kinds_present",
        ) if k in s}
        for s in ov.get("shards", [])
    ],
    "cost_usd_est": {"vm": ${COST_VM}, "disk": ${COST_DISK}, "total": round(${COST_VM}+${COST_DISK}, 4)},
}
Path("${OUT}/run_report.json").write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
b = storage.Client().bucket("authorship-verification-poliphysics")
dv = rep["data_version"]
b.blob(f"tema1-ml/processed/{dv}/run_report.json").upload_from_filename("${OUT}/run_report.json")
print("DONE")
PY
