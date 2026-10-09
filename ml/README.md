# Ballooning surrogate ML (Tema 1)

The researcher only writes model architectures under `ml/models/`. Everything else — data, graphs, training, validation, testing, metrics, logging, cloud helpers — lives here.

## Setup

```bash
cd /path/to/PoliPhysics
python3 -m venv ml/.venv
source ml/.venv/bin/activate
pip install -U pip
# PyTorch: pick the index for your platform (MPS/CUDA/CPU)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -r ml/requirements.txt
pip install -e .   # if using a root pyproject; else:
export PYTHONPATH="/path/to/PoliPhysics:${PYTHONPATH}"
```

Do **not** install `torch_scatter` / `torch_sparse` (optional compiled extensions); the stack uses pure PyG aggregations so it runs on Mac MPS and CUDA.

## Data versions

- Raw Stage B HDF5: `gs://authorship-verification-poliphysics/tema1-stageb/shard-{0,1}/ml_samples/`
- Processed: `gs://authorship-verification-poliphysics/tema1-ml/processed/<data_version>/`
- `data_version` = content hash written in `manifest.json` / `DATA_VERSION`
- Target (**check B**): velocity-based  
  `a_t = (v_{t+1} - v_t) / dt`, `dt = 0.01 s`  
  Store `x_rel` (float32), `spider_abs` (float64), `v`, `u` (float32).
- Frame mask (**check A**): float32 storage makes raw `|dt-0.01|>1e-6` common on long runs; after `round(t/0.01)` all intervals are consecutive, so we mask only true index jumps (≈0%). Rollouts use fixed `dt=0.01` and compare to GT at stored times with linear interpolation on irregular frames only.

### Preprocess

```bash
# splits first (full valid index)
python -m ml.scripts.build_splits --config ml/configs/splits_default.yaml

# local stratified subset (default 5%)
python -m ml.data.preprocess --out "Tema 1/results/ml/processed/local_subset" \
  --subset 0.05 --splits "Tema 1/results/ml/splits/splits.json" --workers 4

# full dataset + upload (usually on a VM)
python -m ml.data.preprocess --out /data/processed --splits ... --upload-gcs
```

## Splits

Configured in `ml/configs/splits_default.yaml` (run-level, never by frame):

| Split | Rule |
|-------|------|
| train / val | `N∈{1,2,4}`, `σ_w∈{0.15,0.30}`, main; val=10% stratified |
| test_interp | 10% held-out from the same grid |
| test_unseenN | all valid `N∈{6,8}` |
| test_unseenSigma | all valid `σ_w=0.22` |

Adding `N=12/16` later = new data + YAML edit only.

## Model interface

```python
from ml.models.base import SurrogateModel, register_model

@register_model("my_model")
class MyModel(SurrogateModel):
    def required_edges(self) -> list[str]:
        return ["material"]  # also: radius, all_tips, fully_connected

    def forward(self, batch):
        # return normalized acceleration [num_nodes, 3]
        ...
```

**How to add a model:** one new file in `ml/models/` + one YAML under `ml/configs/`. Import the module from `ml/models/__init__.py` (one line). Nothing else changes.

Templates: `stub_gnn.py`. Baselines: `constant_velocity`, `follow_air`.

## Train / evaluate / test / tables

```bash
python -m ml.train --config ml/configs/stub_gnn.yaml --device mps
python -m ml.evaluate --config ml/configs/stub_gnn.yaml \
  --checkpoint "Tema 1/results/ml/runs/<run_id>/best.pt" --device mps
python -m ml.test --config ml/configs/stub_gnn.yaml \
  --checkpoint "Tema 1/results/ml/runs/<run_id>/best.pt"
python -m ml.make_tables
```

Every run records git `commit` + `dirty` in `run_meta.json` and W&B (`poliphysics-ballooning-ml`, offline fallback).

Frame sampling default: **equal weight per run** (`sample_weight: per_run`); set `per_frame` for uniform frames.

Surrogate rollouts evaluate air velocity via the simulator’s `KinematicSimulation` (seed + params); that CPU time is included in speed-up / Q* and reported separately as `air_time_per_run`.

## Cloud

```bash
python -m ml.cloud.propose_vm   # prints GPU/cost proposal — creates nothing
# After explicit OK: create VM, rsync processed data, train with resume, then:
bash ml/cloud/sync_run.sh <run_id>
```

Never create/resize/delete VMs, disks, or buckets without an explicit OK.

## Tests

```bash
PYTHONPATH=. pytest ml/tests -q
```
