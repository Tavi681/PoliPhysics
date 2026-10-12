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
- Optional `u_next` (GNS baseline): air at **current** `x_t` and time `t+dt` via `KinematicSimulation`  
  (not `/u_air[t+1]`, which leaks `x_{t+1}`). Present in processed version `5c60fa669a719404`  
  (`store_u_next: true`, `shard_format: consolidated_v1`); on-the-fly compute if missing.
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
| test_unseenN_interp | all valid `N=3` (unseen N, interpolation) |
| test_unseenN | all valid `N∈{6,8}` (unseen N, extrapolation) |
| test_unseenSigma | all valid `σ_w=0.22` |

Adding `N=12/16` later = new data + YAML edit only.

## Model interface

**Worked example — MeshGraphNet / GNS** (`ml/models/gns_baseline.py`):

```python
from ml.models.base import SurrogateModel, register_model

@register_model("gns_baseline")  # also registered as gns_local
class MeshGraphNet(SurrogateModel):
    def required_edges(self) -> list[str]:
        return ["material", "interaction"]  # or ["material"] for local ablation

    def forward(self, batch):
        # Encoder MLPs → M unshared MP steps (residuals) → decoder → â [N,3]
        ...
```

Configs: `ml/configs/gns_baseline.yaml` (material + interaction, uses `u_next`),  
`ml/configs/gns_local.yaml` (material only, no `u_next`). Node/edge feature lists and
globals are switchable in the YAML; the integrator / splits / metrics stay untouched.

**Interaction edges:** radius graph across *different* threads (spider excluded — it is
already linked by material edges) plus all-tips. For `N=1` the interaction set is empty.

**Training noise:** velocity RW with std at the latest step from `norm_stats`
(never from the deleted legacy `noise_stats.json`, which sampled t<5 rest→air
transients and was ~6× too large). Config `noise_mode`:
- `per_type` (default): `σ_{type,c} = noise_k * a_std_type[c] * dt`
- `global` (ablation): `σ_c = noise_k * a_std[c] * dt` for all nodes

`a_by_type` is stored next to global `a` in `norm_stats.json` (same Welford /
`window_ok` pass). Target normalization stays **global**
`(a - a_mean) / a_std`. Target uses clean next velocity:
`a = (v_{t+1}^{gt} - v_t^{noisy}) / dt`. At `k=1`, added normalized variance
is `(a_std_type[c] / a_std[c])^2` per type (or `1` in global mode).
Increments use `σ/√C` then `cumsum`. Default `noise_k: 1`.
k-sweep configs (run later): `ml/configs/sweeps/gns_baseline_k{0.1,0.3,1,3}.yaml`;
select on **validation rollout MSE**, not one-step loss.

**1h sanity (GPU VM):** `ml.scripts.sanity_check_cpu` runs with
`CUDA_VISIBLE_DEVICES=""` on a **copy** of `best.pt` (never a second model on
the training GPU). Exit **44 = SANITY_FAIL** (overall clean loss not below
zero-acc; per-type is logged only) stops training; any other non-zero =
**SANITY_ERROR** (exception / OOM / timeout) is logged and training continues.

**How to add a model:** one new file in `ml/models/` + one YAML under `ml/configs/`.  
Import the module from `ml/models/__init__.py` (one line). Nothing else changes.

Also: `stub_gnn.py`. Zero-param baselines: `constant_velocity`, `follow_air`.

## Train / evaluate / test / tables

```bash
# regenerate local_subset from 5c60fa (same 27 runs, stored u_next):
python -m ml.data.extract_subset \
  --src-uri gs://authorship-verification-poliphysics/tema1-ml/processed/5c60fa669a719404 \
  --run-ids-from "Tema 1/results/ml/processed/local_subset/manifest.json" \
  --out "Tema 1/results/ml/processed/local_subset" --compute-noise-stats

python -m ml.train --config ml/configs/gns_local.yaml --device mps
python -m ml.train --config ml/configs/gns_baseline.yaml --device mps
python -m ml.evaluate --config ml/configs/gns_local.yaml \
  --checkpoint "Tema 1/results/ml/runs/<run_id>/best.pt" --device mps
python -m ml.test --config ml/configs/gns_local.yaml \
  --checkpoint "Tema 1/results/ml/runs/<run_id>/best.pt"
python -m ml.make_tables
```

**Rollout MSE** (`train.rollout_mse` / tables): mean spider-relative **position** error in
**m²**, averaged over nodes, over each of `K` integrated steps (`K = rollout_K`, default
20 → horizon `0.20 s` at `dt=0.01`), and over up to `max_runs` runs. Full evaluation also
reports per test split `|ΔP|` and relative `|Δ(R/L)|` from complete surrogate rollouts.

Every run records git `commit` + `dirty` in `run_meta.json` and W&B (`poliphysics-ballooning-ml`, offline fallback).

Frame sampling default: **equal weight per run** (`sample_weight: per_run`); set `per_frame` for uniform frames.

Surrogate rollouts evaluate air velocity with a **batched torch** port of the simulator’s
`KinematicSimulation` (modes from HDF5 `/turb` when available, else the same deterministic
`make_ks(seed,…)` stream). That air time is still counted in speed-up / Q* as
`air_time_per_run`.

**Tiered validation (GPU configs):** every `val_every` steps — one-step MSE on a fixed
sample of val frames; every `val_rollout_every` (auto-sized so rollout ≤10% wall) —
batched rollouts on a stratified subset of val runs (checkpoint metric). Full test-split
eval only at the end of training.

## Cloud

Default **training** VM (create only when starting a full train run): `g2-standard-8` + 1× L4, us-central1, on-demand. If W&B shows high `train/data_wait_s` and low `train/gpu_util`, use documented `g2-standard-16` + 1× L4.

```bash
python -m ml.cloud.propose_vm   # prints defaults — creates nothing
# Full preprocess (CPU VM, stream from GCS):
bash ml/cloud/run_preprocess_on_gcp.sh
# Full train (L4; does not delete VM / b0a7):
bash ml/cloud/run_train_on_gcp.sh
# Sync a finished run:
bash ml/cloud/sync_run.sh <run_id>
```

Never create/resize/delete VMs, disks, or buckets without an explicit OK (preprocess VM OK is this task).

## Tests

```bash
PYTHONPATH=. pytest ml/tests -q
```
