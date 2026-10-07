# Clamped-release check

## 1. How the clamp phase decides “steady”

From `integrator.py` (Alg.1 lines 18–19), used by phase 1 of `_simulate_clamped_release`:

| Item | Value |
|---|---|
| Quantity | Spider vertical velocity `zdot_0 = ξ̇[2]` |
| Test | `\|zdot₀ⁿ − zdot₀ⁿ⁻¹\| / max(\|zdot₀ⁿ\|, 1) < δ` |
| Tolerance `δ` | `1e-6` (`sweep_params` / `Params.delta`) |
| Minimum duration | Continuous satisfaction for `t_w = 0.05` s |
| Maximum duration | Parent `t_end` (`SWEEP_T_END = 60` s) |
| Phase-1 flags | `stop_on_steady=True`, `use_alg2_stopping=False`, `ZeroFlow`, Dirichlet `fixed_dofs=[0,1,2]` |

### Actual clamp duration (N = 2, 4, 8; σ_w = 0.30, x = 0)

All three stop at **t = 0.051 s** with `status=steady`, R/L equal to the sweep frame-0 values:

| N | phase1_t_exit (s) | R/L at stop |
|---|-------------------|-------------|
| 2 | 0.051 | 6.587e−02 |
| 4 | 0.051 | 7.295e−02 |
| 8 | 0.051 | 7.388e−02 |

**Why so short:** with the spider fixed, `zdot_0 ≡ 0`, so the relative-change test is always true (denom falls back to 1). Steady is declared after the first `t_w = 0.05` s window — long before electrostatic opening finishes.

## 2. Still-air clamp for 10 s (F̄_l = 1, N_t = 50)

`q = mg/(N E)`, spider fixed, `stop_on_steady=False`, `t_end = 10` s.

| N | R/L (10 s) | t₉₉% (s) | (S_N Λ/4)^(1/3) | err % | frame 0 (sweep) |
|---|------------|----------|-----------------|-------|-----------------|
| 2 | 9.284e−02 | 0.58 | 9.294e−02 | 0.12 | 6.587e−02 |
| 4 | 1.152e−01 | 1.12 | 1.154e−01 | 0.20 | 7.295e−02 |
| 8 | 1.268e−01 | 1.38 | 1.311e−01 | 3.24 | 7.388e−02 |

Series: `clamp_10s_series.csv`. Asymptote ≫ frame 0 → production clamp is **not** fully relaxed.

## 3. Re-release from fully relaxed clamp

Running separately under `results/clamp_check/` (does not touch `sweep.csv` / `ml_samples/`).
Grid: N ∈ {4,8} × σ_w = 0.30 × x ∈ {0,1} × M = 200 (same seeds). Progress in `part3.log`; summary → `relaxed_vs_sweep.csv` when done.

## 4. fig_snapshot

Written to `results/figs/fig_snapshot.png` (and `results/fig_snapshot.png`).
