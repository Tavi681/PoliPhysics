"""1h sanity check that never touches the training GPU.

Forces CUDA_VISIBLE_DEVICES="" before importing torch CUDA paths via diagnose.
Copies the checkpoint so the training process's file is not mmap'd for the check.

Exit codes:
  0  — SANITY_OK (clean overall loss below zero-acc; per-type logged only)
  44 — SANITY_FAIL (overall clean loss not below zero-acc)
  45 — SANITY_ERROR (exception / timeout / unexpected failure — do NOT stop train)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path


def main() -> int:
    # Hide every CUDA device before any torch CUDA init in diagnose_loss_refs.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Prefer CPU even if MPS is present when --device cpu (default).
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout-s", type=float, default=1800.0,
                    help="soft note only; hard timeout left to the caller (timeout(1))")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"SANITY_ERROR missing checkpoint {ckpt}", flush=True)
        return 45

    tmp_dir = Path(tempfile.mkdtemp(prefix="sanity_ckpt_"))
    ckpt_copy = tmp_dir / "best_copy.pt"
    try:
        shutil.copy2(ckpt, ckpt_copy)
        # Import after CUDA_VISIBLE_DEVICES is cleared.
        from ml.scripts import diagnose_loss_refs as dlr

        sys.argv = [
            "diagnose_loss_refs",
            "--config", str(args.config),
            "--checkpoint", str(ckpt_copy),
            "--device", "cpu",
            "--out", str(args.out),
            "--n-batches", str(args.n_batches),
            "--seed", str(args.seed),
            "--require-below-zero-acc",
        ]
        try:
            dlr.main()
        except SystemExit as e:
            code = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 45)
            if code == 44:
                print("SANITY_FAIL", flush=True)
                return 44
            if code == 0:
                print("SANITY_OK", flush=True)
                return 0
            print(f"SANITY_ERROR diagnose exited {code}", flush=True)
            return 45
        print("SANITY_OK", flush=True)
        return 0
    except Exception as e:
        print(f"SANITY_ERROR {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return 45
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
