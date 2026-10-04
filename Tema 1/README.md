# Tema 1 — Charged ballooning spider (DER, Algorithm 1)

Implicit Discrete Elastic Rods time stepping for a multi-thread charged
ballooning spider (Habchi & Jawed / Bergou DER), matching
[`tema1_objective.md`](tema1_objective.md).

## Setup

From this directory (`Tema 1/`):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[test]"
pip install pandas matplotlib      # only needed for plot_results.py
```

Python ≥ 3.11. Core deps: NumPy, SciPy, h5py (see `pyproject.toml`).

## Package layout

| Path | Role |
|---|---|
| `ballooning/` | Simulator: geometry, frames, elastic energy, forces, integrator (Alg. 1), observables, HDF5 I/O, CLI |
| `ballooning/studies.py` | Named validation / production studies shared by tests and scripts |
| `tests/` | Pytest suite (fast FD checks + `@pytest.mark.slow` physics runs) |
| `scripts/` | Production and invariance pipelines |
| `examples/` | Small demos |
| `docs/algorithm_mapping.md` | Traceability: Alg. 1 lines ↔ code, equation labels |
| `results/` | CSV tables/figures data, `meta.json`, optional PDFs under `results/figs/` |
| `plot_results.py` | Build paper-style PDFs from `results/*.csv` |

## Scripts

All commands below are run **from `Tema 1/`** with the venv active.

### `scripts/produce_results.py`

Runs the production studies (terminal velocity, chamber 4a/4b, convergence,
collapse / steady / shapes / invariant / flow invariance) and writes

`results/tab_*.csv`, `results/fig_*.csv`, `results/meta.json`, `results/log.txt`.

```bash
python scripts/produce_results.py
python scripts/produce_results.py --workers 8   # optional process-pool size
```

Convergence cases run sequentially (clean `cpu_s`); other jobs run in parallel.
Wall time is on the order of several minutes on Apple M-series (N≤8, N_t=100).

### `scripts/run_invariance_equal_t.py`

Fixed-`dt` equal-time Galilean / uniform-flow invariance check (writes into
`results/meta.json` under `studies.fig_invariance.equal_t`):

- **comoving:** w-run starts with every node velocity = `w ẑ` (at rest relative
  to the air); w=0 keeps `v0=0`. Expect max spider-frame deviation / L at Newton
  noise (~1e-11–1e-12 with `dt=3e-5`, `eps=1e-10`).
- **transient:** both runs use `v0=0` (different relative IC); reports decay time
  vs Stokes time `t_s`.

```bash
# Full production check (~3–4 h wall with 3 parallel workers)
python -u scripts/run_invariance_equal_t.py --dt 3e-5 \
  --cache-dir results/equal_t_cache --meta-out results/meta.json

# Rebuild meta from cached trajectories (no re-integration)
python scripts/run_invariance_equal_t.py --from-cache results/equal_t_cache \
  --dt 3e-5 --meta-out results/meta.json

# Short end-to-end smoke (does not overwrite production meta.json)
python scripts/run_invariance_equal_t.py --smoke
```

Useful flags: `--t-end`, `--output-dt`, `--serial`, `--cache-dir`, `--meta-out`.

Default Newton settings are kept (`eps=1e-10`, `K=20`). Fixed `dt=1e-4` and
`5e-5` fail to converge; `3e-5` is the largest candidate that works.

### `scripts/smoke_invariance_equal_t.py`

Fast checks before a multi-hour equal-t run: float time-matching, `dt`
selection, co-moving IC, parallel pipeline, cache, meta write, `--from-cache`.

```bash
python scripts/smoke_invariance_equal_t.py
```

### `scripts/stage_a.py` — turbulence generator, cheap studies, pilot

Stage A adds a synthetic-turbulence flow model (`flow_model="kinematic"`,
Kinematic Simulation, Eq. `eq:ks` with the von Kármán spectrum `eq:vk`) and its
studies. This script is **additive**: it never deletes the baseline
`results/*.csv`; it only writes the Stage A files and merges a `stage_a` section
into `results/meta.json`.

```bash
python scripts/stage_a.py --ksvalid --tab-wc --relax   # cheap studies
python scripts/stage_a.py --pilot --pilot-smoke        # short pilot pipeline check
python scripts/stage_a.py --pilot                      # full pilot (hours; see below)
python scripts/stage_a.py --all
```

Outputs:

| File | Study |
|---|---|
| `fig_ksvalid.csv` | measured vs target 1-D longitudinal spectrum of `w` (`N_k`, `k`, `E_target`, `E_measured`) |
| `fig_ksvalid_pdf.csv` | one-point PDF of `w` vs a Gaussian(σ) |
| `tab_wc.csv` | steady fall speed `w_c` vs analytic `w_s(1-Fbar_l)` over N, m, q (table `tab:wc`) |
| `relax.csv` | lateral relaxation `dR/L`, `dshape/L` from the equal-t transient cache (+ decay times in `meta.json`) |
| `pilot.csv` | Algorithm `alg:sweep` pilot cost per run (outcome, wall/sim, dt stats, Newton failures, max\|u\|) |
| `ml_samples/pilot_N*.h5` | float32 HDF5 samples; bytes / simulated second reported in `meta.json` |

The **pilot** (`--pilot`) runs `N∈{1,2,4,8} × 3 seeds` for `t_end=60 s` with
clamped release into turbulence, in parallel. Cost scales steeply with `N`
(wall/sim ≈ 10 at N=1, ≈ 150 at N=8), so a full pilot is a multi-hour run. Use
`--pilot-tend`/`--pilot-nt` to shrink it, or `--pilot-smoke` to validate the
pipeline (`t_end=0.5`, `N_t=40`). It does **not** start the production sweep.

`release_mode="clamped"` (Alg.2 line 4) holds node 0 fixed at `z0` in still air
until steady, then releases at `t=0` into the configured flow.

### `scripts/stage_b.py` — production sweep (Algorithm `alg:sweep`)

Stage B grid: `N∈{1,2,4,8} × σ_w∈{0.15,0.30} × x∈{-4,-2,-1,0,1,2,4} × M=200`
(= 11 200 runs), clamped release, `turb_renormalize=True`, `N_k=200`. Resume-safe
(`sweep.csv` append; skips completed seeds). Progress every 10 min in
`results/sweep_log.txt`.

```bash
python scripts/stage_b.py --smoke                 # local pipeline check
python scripts/stage_b.py --ksvalid               # renormalized fig_ksvalid (N_k=200)
python scripts/stage_b.py --tab-wc-m10            # m=10 mg rows, t_end=40 s
python scripts/stage_b.py --cost-probe            # 1 point/N × M=5; print ETA
python scripts/stage_b.py --sweep                 # full grid (wait for confirmation)
```

Outputs: `results/sweep.csv`, `results/tab_phase.csv`, `results/snapshots/*.npz`,
`results/ml_samples/*.h5` (gitignored).

### `scripts/run_stage_b_on_gcp.sh` — Stage B on GCP

Same pattern as Tema 2 `run_round6_export_on_gcp.sh`: create VM → scp package →
run → pull results → delete VM (unless `DETACH=true`). Resume-safe via packed
`results/sweep.csv`. **Default `PREEMPTIBLE=false`** (on-demand) because Stage B is a
long run; set `PREEMPTIBLE=true` only if you accept spot preemption.

```bash
# 1) cost probe (default MODE) — prints ETA, then wait for confirmation
bash "Tema 1/scripts/run_stage_b_on_gcp.sh"

# 2) full sweep, leave VM running (recommended for long jobs / laptop close)
MODE=sweep DETACH=true bash "Tema 1/scripts/run_stage_b_on_gcp.sh"

# Optional: MODE=prep|sweep|all|cost-probe
#           MACHINE=c2-standard-16 JOBS=16 PREEMPTIBLE=false DISK_GB=100
#           PULL_ML=true NO_HDF5=true INSTANCE=... ZONE=...
```

Status while detached (running / failed / done):

```bash
bash "Tema 1/scripts/check_stage_b_gcp.sh"
```

With `DETACH=true`, the script prints `gcloud` commands to check progress, pull
`sweep.csv` / `tab_phase.csv` / `snapshots/`, and delete the VM. The VM is kept
even if setup fails (so you can inspect logs). Image: Ubuntu 24.04 (Python ≥ 3.12).

After a `sweep`/`all` finishes (even without `DETACH`), small artifacts are pulled
automatically, then the script prints `du -sh` for `ml_samples/` and **leaves the
VM up** so you can choose: pull locally (`PULL_ML=true` next time, or the printed
`scp`), upload to GCS (`GCS_BUCKET=gs://...` or the printed `gsutil`), then delete
the instance yourself. Cost-probe / prep still auto-delete the VM.

### `scripts/check_regression.py`

Compares `results/*.csv` with `results/pre_stageA/*.csv` (max relative difference
`< 1e-12`, timing columns excluded) and writes the verdict to
`results/regression.txt`. Run `scripts/produce_results.py` first to regenerate
the baseline CSVs with the current code, then this.

### `plot_results.py`

Reads `results/*.csv` and writes PDFs to `results/figs/`:

`fig_test1.pdf`, `fig_collapse.pdf`, `fig_invariant.pdf`, `fig_shapes.pdf`,
`fig_invariance.pdf`.

```bash
python plot_results.py
```

Requires `pandas` and `matplotlib`.

## Examples

```bash
python examples/run_single.py          # small N=2 run → run_single.h5 (gitignored)
python examples/convergence.py         # thin wrapper around the N=2 convergence study
```

CLI entry point after `pip install -e .`:

```bash
ballooning --help
```

## Tests

```bash
pytest -m "not slow" -q          # FD / unit checks (~1 s)
pytest -m slow -q                # physics validations (minutes)
pytest -q                        # everything
```

## Results

`scripts/produce_results.py` is the source of truth for CSV columns (SI unless
the header says otherwise). `results/meta.json` records parameters, environment,
wall-clock times, and the equal-t invariance block.

Large regenerable artifacts are gitignored: `.venv/`, `*.h5`, equal-t trajectory
caches, smoke metas, and run logs. Keep committed: production CSVs, `meta.json`,
and optional `results/figs/*.pdf`.

## Equation labels

Code comments use the paper’s LaTeX labels (`eq:residual`, `eq:Es`, `eq:Eb`,
`eq:Et`, `eq:fext`, `eq:Fvk`, `eq:coulomb`, `eq:jacobian`, and the Stage A
labels `eq:ks`, `eq:vk`, `alg:sweep`). See `docs/algorithm_mapping.md`.
