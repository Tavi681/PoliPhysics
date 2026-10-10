"""Propose (do NOT create) GCP GPU VMs for training / benchmarks.

Requires explicit OK to provision. After OK: create ONE VM at a time, run a
15-minute stub_gnn N=8 benchmark, report steps/s and cost/epoch, then stop and
delete only after a second OK.
"""
from __future__ import annotations

# On-demand USD prices from Google Cloud public pricing pages (default region
# Iowa / us-central1), retrieved 2026-10-10:
# - n1-standard-8: $0.379998/h
#   https://cloud.google.com/products/compute/pricing/general-purpose
# - NVIDIA T4 attach: $0.35/h
#   https://cloud.google.com/products/compute/pricing/accelerator-optimized
#   (GPU Attachment pricing table)
# - g2-standard-8 (1× L4 bundled): $0.853624312/h
#   https://cloud.google.com/products/compute/pricing/accelerator-optimized

PRICING_URLS = {
    "general_purpose": "https://cloud.google.com/products/compute/pricing/general-purpose",
    "accelerator_optimized": "https://cloud.google.com/products/compute/pricing/accelerator-optimized",
    "gpu_docs": "https://cloud.google.com/compute/docs/gpus",
}

OPTIONS = [
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
    {
        "name": "L4",
        "machine_type": "g2-standard-8",
        "gpu": "nvidia-l4 (bundled)",
        "gpu_count": 1,
        "gpu_memory_gb": 24,
        "vcpus": 8,
        "ram_gb": 32,
        "region": "us-central1",
        "on_demand_usd_per_hour": {
            "total": 0.853624312,
        },
        "source": PRICING_URLS["accelerator_optimized"],
    },
]

BENCHMARK_PLAN = {
    "duration_min": 15,
    "workload": "stub_gnn forward+backward on full-size N=8 graphs (batch)",
    "report": ["steps/s", "cost per epoch = (steps_per_epoch / steps_per_s) * $/s"],
    "rule": "Create ONE VM at a time after OK; delete only after OK.",
}

PREPROCESS_PLAN = {
    "where": "CPU VM in the same region as the bucket (us-central1), not the Mac",
    "bucket": "gs://authorship-verification-poliphysics/tema1-stageb/",
    "processed_prefix": "gs://authorship-verification-poliphysics/tema1-ml/processed/<data_version>/",
    "suggested_machine": "c2-standard-16 (or n2-standard-16); no GPU",
    "machine_usd_per_hour_est": 0.6688,  # c2-standard-16 us-central1 ballpark; confirm on pricing page
    "modes": {
        "stream_from_gcs": {
            "description": (
                "Read each HDF5 via gcsfs+h5py (or gsutil cat) on the VM; write only "
                "processed shards locally then upload. No full raw copy."
            ),
            "disk_gb": {"processed": 52, "margin": 30, "total_pd_ssd": 100},
            "estimated_time_h": "3–6 h (8200 runs, ~16–32 workers; network-bound)",
            "estimated_cost_usd": "~$3–8 VM + ~$1–3 disk pro-rated",
        },
        "copy_raw_then_process": {
            "description": "gsutil -m rsync raw ml_samples to PD, then preprocess, then upload processed.",
            "disk_gb": {
                "raw_hdf5": 59,
                "processed": 52,
                "margin": 40,
                "total_pd_ssd": 160,
            },
            "estimated_time_h": "4–8 h wall (download + convert 8200 runs)",
            "estimated_cost_usd": "~$5–15 VM + disk (~$0.17/GB-mo pd-ssd, pro-rated)",
        },
    },
    "preferred_mode": "stream_from_gcs",
    "command_sketch": (
        "python -m ml.data.preprocess --out /data/processed "
        "--splits ... --workers 16 --upload-gcs"
    ),
    "do_not_start_before_ok": True,
}


def main():
    import json
    print(json.dumps({
        "gpu_options": OPTIONS,
        "pricing_urls": PRICING_URLS,
        "benchmark_plan": BENCHMARK_PLAN,
        "preprocess_plan": PREPROCESS_PLAN,
        "note": "No resources created. Reply OK with option name (T4 or L4) to create ONE benchmark VM.",
    }, indent=2))


if __name__ == "__main__":
    main()
