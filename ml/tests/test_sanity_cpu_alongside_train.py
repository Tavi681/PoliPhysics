"""Sanity check on CPU must not change an active MPS training process's memory."""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def _train_worker(ready: mp.Queue, stop: mp.Queue, mem_q: mp.Queue) -> None:
    """Hold a non-trivial MPS allocation and run a tiny training loop."""
    import torch as t
    import torch.nn as nn

    assert t.backends.mps.is_available()
    device = t.device("mps")
    # ~64 MB buffer that stays allocated for the duration of the check.
    hold = t.zeros((16, 1024, 1024), device=device, dtype=t.float32)
    model = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 64)).to(device)
    opt = t.optim.Adam(model.parameters(), lr=1e-3)
    x = t.randn(32, 64, device=device)
    y = t.randn(32, 64, device=device)

    def _alloc() -> int:
        if hasattr(t.mps, "current_allocated_memory"):
            return int(t.mps.current_allocated_memory())
        return int(hold.nelement() * hold.element_size())

    # Warmup
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    if hasattr(t.mps, "synchronize"):
        t.mps.synchronize()
    mem0 = _alloc()
    mem_q.put(mem0)
    ready.put("ready")
    # Keep training until stop
    while stop.empty():
        opt.zero_grad(set_to_none=True)
        loss = (model(x) - y).pow(2).mean() + 1e-8 * hold.mean()
        loss.backward()
        opt.step()
        time.sleep(0.01)
    if hasattr(t.mps, "synchronize"):
        t.mps.synchronize()
    mem_q.put(_alloc())


@pytest.mark.skipif(not _mps_available(), reason="MPS required")
def test_sanity_cpu_does_not_change_mps_train_memory(tmp_path: Path):
    """While an MPS train process is active, CPU sanity must not move its MPS alloc."""
    ctx = mp.get_context("spawn")
    ready: mp.Queue = ctx.Queue()
    stop: mp.Queue = ctx.Queue()
    mem_q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_train_worker, args=(ready, stop, mem_q), daemon=True)
    proc.start()
    assert ready.get(timeout=60) == "ready"
    mem_before = int(mem_q.get(timeout=30))
    assert mem_before > 10_000_000  # held buffer ~64MB

    # Run sanity wrapper with forced empty CUDA devices; device=cpu path.
    # Use a tiny synthetic checkpoint path that will SANITY_ERROR (missing), which
    # still exercises CUDA_VISIBLE_DEVICES="" and the CPU import path without
    # needing full ballooning data. Then also call resolve path that only uses CPU.
    env_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        from ml.scripts.sanity_check_cpu import main as sanity_main
        import sys

        missing = tmp_path / "nope.pt"
        sys.argv = [
            "sanity_check_cpu",
            "--config", "ml/configs/gns_local_mps_verify.yaml",
            "--checkpoint", str(missing),
            "--out", str(tmp_path / "out.json"),
        ]
        rc = sanity_main()
        assert rc == 45  # SANITY_ERROR on missing ckpt — must not touch MPS
    finally:
        if env_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = env_cuda

    # Also verify a real CPU torch op with CUDA hidden doesn't allocate on MPS
    # from this process (sanity runs in-process here).
    if hasattr(torch.mps, "current_allocated_memory"):
        # This process should have ~0 MPS alloc (train is another process).
        here = int(torch.mps.current_allocated_memory())
        assert here < 1_000_000, f"sanity process leaked MPS memory: {here}"

    time.sleep(0.5)
    stop.put("stop")
    proc.join(timeout=30)
    assert not proc.is_alive()
    mem_after = int(mem_q.get(timeout=30))
    # Training process memory must stay within 5% (no second model on MPS).
    rel = abs(mem_after - mem_before) / max(mem_before, 1)
    assert rel < 0.05, f"MPS train mem changed: before={mem_before} after={mem_after} rel={rel}"


@pytest.mark.skipif(not _mps_available(), reason="MPS required")
def test_sanity_check_cpu_forces_empty_cuda_visible_devices(tmp_path: Path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from ml.scripts.sanity_check_cpu import main as sanity_main
    import sys

    sys.argv = [
        "sanity_check_cpu",
        "--config", "ml/configs/gns_local_mps_verify.yaml",
        "--checkpoint", str(tmp_path / "missing.pt"),
        "--out", str(tmp_path / "o.json"),
    ]
    rc = sanity_main()
    assert rc == 45
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
