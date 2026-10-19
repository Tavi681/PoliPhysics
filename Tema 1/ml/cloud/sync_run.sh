#!/usr/bin/env bash
# Sync a local run directory to GCS. Does not create buckets.
set -euo pipefail
RUN_ID="${1:?run_id}"
BUCKET="${BUCKET:-gs://authorship-verification-poliphysics/tema1-ml/runs}"
SRC="${2:-results/ml/${RUN_ID}}"
echo "Syncing ${SRC} → ${BUCKET}/${RUN_ID}/"
gsutil -m rsync -r "${SRC}" "${BUCKET}/${RUN_ID}/"
