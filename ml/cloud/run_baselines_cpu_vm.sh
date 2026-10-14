#!/usr/bin/env bash
# Reference-method (CV + follow-air) full paper eval on a CPU VM — orchestration hints only.
# Caller creates/starts the VM, mounts /data/processed/5c60fa669a719404, clones repo.
#
# On the VM (after ssh):
#   cd ~/poliphysics && export PYTHONPATH=.
#
# Build throwaway baseline checkpoints (no training):
#   python3 - <<'PY'
# from pathlib import Path
# import torch
# from ml.utils.config import load_yaml
# from ml.train import _resolve
# from ml.data.dataset import BallooningRunStore
# from ml.models import build_model
# def make(name, cfg_path, out_name):
#     cfg = load_yaml(_resolve(cfg_path))
#     dd = cfg["data"]["data_dir"]
#     store = BallooningRunStore(dd if Path(dd).is_absolute() else _resolve(dd))
#     m = build_model(name, a_mean=store.norm["a"]["mean"], a_std=store.norm["a"]["std"])
#     out = Path("Tema 1/results/ml/runs") / out_name
#     out.mkdir(parents=True, exist_ok=True)
#     torch.save({"model": m.state_dict(), "meta": {"run_id": out.name, "train_time_s": 0.0}, "cfg": cfg}, out / "best.pt")
#     print(out / "best.pt")
# make("constant_velocity", "ml/configs/baseline_const_cpu.yaml", "_baseline_const_cpu_paper")
# make("follow_air", "ml/configs/baseline_follow_air_cpu.yaml", "_baseline_follow_air_cpu_paper")
# PY
#
# Full eval (streaming rollouts, resume per run under eval output dir):
#   python3 -m ml.evaluate --config ml/configs/baseline_const_cpu.yaml \
#     --checkpoint "Tema 1/results/ml/runs/_baseline_const_cpu_paper/best.pt" \
#     --device cpu --splits test_interp test_unseenN_interp test_unseenN test_unseenSigma
#   python3 -m ml.evaluate --config ml/configs/baseline_follow_air_cpu.yaml \
#     --checkpoint "Tema 1/results/ml/runs/_baseline_follow_air_cpu_paper/best.pt" \
#     --device cpu --splits test_interp test_unseenN_interp test_unseenN test_unseenSigma
#
# Sync to GCS (set BUCKET):
#   BUCKET=gs://authorship-verification-poliphysics/tema1-ml
#   for RID in _baseline_const_cpu_paper _baseline_follow_air_cpu_paper; do
#     gsutil -m rsync -r "Tema 1/results/ml/${RID}" "${BUCKET}/runs/${RID}/"
#     gsutil -m rsync -r "Tema 1/results/ml/runs/${RID}" "${BUCKET}/runs/${RID}/ckpt/"
#   done
#
set -euo pipefail
echo "run_baselines_cpu_vm.sh: documentation-only; run commands above on the CPU VM."
