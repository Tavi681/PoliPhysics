"""Propose (do NOT create) GCP GPU VMs for training / benchmarks.

Default training target: g2-standard-8 + 1× L4 (us-central1, on-demand).
Create the GPU VM only when starting a full training run (explicit OK).
"""
from __future__ import annotations

# On-demand USD (us-central1), from Google Cloud pricing pages (2026-10-10):
# https://cloud.google.com/products/compute/pricing/accelerator-optimized
# https://cloud.google.com/products/compute/pricing/general-purpose

PRICING_URLS = {
    "general_purpose": "https://cloud.google.com/products/compute/pricing/general-purpose",
    "accelerator_optimized": "https://cloud.google.com/products/compute/pricing/accelerator-optimized",
    "gpu_docs": "https://cloud.google.com/compute/docs/gpus",
}

# Default for full training (do not auto-create)
DEFAULT_TRAIN_VM = {
    "name": "L4",
    "machine_type": "g2-standard-8",
    "gpu": "nvidia-l4 (bundled)",
    "gpu_count": 1,
    "gpu_memory_gb": 24,
    "vcpus": 8,
    "ram_gb": 32,
    "region": "us-central1",
    "zone": "us-central1-a",
    "on_demand_usd_per_hour": {"total": 0.853624312},
    "source": PRICING_URLS["accelerator_optimized"],
}

# Use if training is dataloader / CPU-bound (more host cores)
LOADER_BOUND_OPTION = {
    "name": "L4_x16",
    "machine_type": "g2-standard-16",
    "gpu": "nvidia-l4 (bundled)",
    "gpu_count": 1,
    "gpu_memory_gb": 24,
    "vcpus": 16,
    "ram_gb": 64,
    "region": "us-central1",
    "note": "Documented option when W&B shows high dataloader wait vs GPU util",
    "on_demand_usd_per_hour": {"total": 1.0044},  # confirm on pricing page before create
    "source": PRICING_URLS["accelerator_optimized"],
}

OPTIONS = [
    DEFAULT_TRAIN_VM,
    LOADER_BOUND_OPTION,
    {
        "name": "T4",
        "machine_type": "n1-standard-8",
        "gpu": "nvidia-tesla-t4",
        "gpu_count": 1,
        "gpu_memory_gb": 16,
        "vcpus": 8,
        "ram_gb": 30,
        "region": "us-central1",
        "on_demand_usd_per_hour": {
            "vm": 0.379998,
            "gpu": 0.35,
            "total": 0.729998,
        },
        "source": PRICING_URLS["general_purpose"] + " + " + PRICING_URLS["accelerator_optimized"],
    },
]

PREPROCESS_VM = {
    "machine_type": "c2-standard-16",
    "region": "us-central1",
    "zone": "us-central1-a",
    "disk_gb": 100,
    "disk_type": "pd-balanced",
    "mode": "stream_from_gcs",
    "on_demand_usd_per_hour": 0.6688,  # c2-standard-16 us-central1 ballpark
}


def main():
    import json
    print(json.dumps({
        "default_train_vm": DEFAULT_TRAIN_VM,
        "loader_bound_option": LOADER_BOUND_OPTION,
        "gpu_options": OPTIONS,
        "preprocess_vm": PREPROCESS_VM,
        "pricing_urls": PRICING_URLS,
        "note": (
            "Default train VM is g2-standard-8 + 1x L4. Do NOT create it until "
            "a full training run is started. Watch W&B train/gpu_util and "
            "train/data_wait_s for loader bottlenecks; escalate to g2-standard-16 if needed."
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
