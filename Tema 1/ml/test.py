"""CLI alias: evaluate all three test splits from a checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

from ml.evaluate import evaluate_checkpoint
from ml.train import _device, _resolve
from ml.utils.config import load_yaml
from ml.utils.git_info import repo_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    cfg_path = args.config if args.config.is_absolute() else _resolve(args.config)
    if not cfg_path.exists():
        cfg_path = repo_root() / "ml" / args.config
    cfg = load_yaml(cfg_path)
    evaluate_checkpoint(
        Path(args.checkpoint),
        cfg,
        _device(args.device),
        ["test_interp", "test_unseenN_interp", "test_unseenN", "test_unseenSigma"],
    )


if __name__ == "__main__":
    main()
