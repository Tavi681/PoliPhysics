"""Propose (do NOT create) a GCP GPU VM for training. Requires explicit OK to provision."""
from __future__ import annotations

PROPOSAL = {
    "machine_type": "n1-standard-8",
    "gpu": "nvidia-tesla-t4",  # 16 GB; good price/perf for GNN training
    "gpu_count": 1,
    "disk": "200GB pd-ssd (persistent)",
    "image": "pytorch / deep learning VM or ubuntu-2204 + CUDA 12",
    "region": "us-central1 (or your quota region)",
    "on_demand_usd_per_hour_approx": {
        "n1-standard-8": 0.38,
        "nvidia-tesla-t4": 0.35,
        "total_approx": 0.73,
        "note": "Check current GCP pricing; on-demand not spot as requested.",
    },
    "alt_gpu": {
        "gpu": "nvidia-l4",
        "reason": "Newer, often better for PyTorch 2.x; check regional availability/price",
    },
    "actions_requiring_ok": [
        "gcloud compute instances create",
        "disk create/resize/delete",
        "bucket create/delete",
    ],
    "workflow_after_ok": [
        "1. create VM + attach persistent disk",
        "2. mount disk, clone repo, create ml/.venv from ml/requirements.txt",
        "3. gsutil -m rsync processed data_version from gs://.../tema1-ml/processed/<ver>/",
        "4. python -m ml.train --config ... --device cuda",
        "5. sync checkpoints/results to gs://.../tema1-ml/runs/<run_id>/",
    ],
}


def main():
    import json
    print(json.dumps(PROPOSAL, indent=2))
    print("\n*** No resources created. Reply OK with machine/GPU choice to provision. ***")


if __name__ == "__main__":
    main()
