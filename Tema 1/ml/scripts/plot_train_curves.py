"""Plot train loss / val one-step / val rollout vs steps from a train log.

  PYTHONPATH=. python -m ml.scripts.plot_train_curves \\
    --log "results/ml/logs/gns_local_gpu_train.log" \\
    --refs "results/ml/logs/rollout_refs.json" \\
    --out "results/ml/runs/<run_id>/fig_curves.png"
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ml.train import _resolve


STEP_RE = re.compile(
    r"^STEP\s+(?P<step>\d+)\s+.*\bloss=(?P<loss>[-+eE0-9.]+)",
)
VAL_RE = re.compile(
    r"^step\s+(?P<step>\d+):\s+train_loss=(?P<train>[-+eE0-9.]+)\s+"
    r"val_1step=(?P<v1>[-+eE0-9.]+)\s+val_rollout=(?P<vro>[-+eE0-9.]+)",
)


def parse_log(path: Path):
    steps, losses = [], []
    v_steps, v_train, v_1step, v_roll = [], [], [], []
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.match(line.strip())
        if m:
            steps.append(int(m.group("step")))
            losses.append(float(m.group("loss")))
            continue
        m = VAL_RE.match(line.strip())
        if m:
            v_steps.append(int(m.group("step")))
            v_train.append(float(m.group("train")))
            v_1step.append(float(m.group("v1")))
            v_roll.append(float(m.group("vro")))
    return {
        "train_steps": np.asarray(steps),
        "train_loss": np.asarray(losses),
        "val_steps": np.asarray(v_steps),
        "val_train": np.asarray(v_train),
        "val_1step": np.asarray(v_1step),
        "val_rollout": np.asarray(v_roll),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--refs", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    log = _resolve(args.log) if not args.log.is_absolute() else args.log
    out = _resolve(args.out) if not args.out.is_absolute() else args.out
    data = parse_log(log)
    refs = {}
    if args.refs is not None:
        rp = _resolve(args.refs) if not args.refs.is_absolute() else args.refs
        if rp.exists():
            refs = json.loads(rp.read_text())

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    ax0, ax1, ax2 = axes
    if len(data["train_steps"]):
        # downsample for readability
        s = data["train_steps"]
        y = data["train_loss"]
        if len(s) > 2000:
            idx = np.linspace(0, len(s) - 1, 2000, dtype=int)
            s, y = s[idx], y[idx]
        ax0.plot(s, y, lw=0.8, color="C0", label="train loss (batch)")
    ax0.set_ylabel("train loss")
    ax0.set_yscale("log")
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.3)

    if len(data["val_steps"]):
        ax1.plot(data["val_steps"], data["val_1step"], "o-", ms=3, label="val one-step")
    ax1.set_ylabel("val one-step MSE")
    ax1.set_yscale("log")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    if len(data["val_steps"]):
        ax2.plot(data["val_steps"], data["val_rollout"], "o-", ms=3, color="C2", label="val rollout")
    if "constant_velocity" in refs:
        ax2.axhline(float(refs["constant_velocity"]), color="C3", ls="--", label="constant_velocity")
    if "follow_air" in refs:
        ax2.axhline(float(refs["follow_air"]), color="C4", ls="--", label="follow_air")
    ax2.set_ylabel("val rollout MSE")
    ax2.set_xlabel("step")
    ax2.set_yscale("log")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
