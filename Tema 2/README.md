# netsim - drone impact on a prestressed net (Algorithm 1)

Explicit velocity-Verlet mass-spring simulator for the impact of a rigid drone
on a prestressed planar net, with permanent thread failure. Implements
Algorithm 1 of `net_students_en.tex` (Section "Numerical method"). All units SI.

## Installation

```bash
cd "Tema 2"
python -m pip install -e .            # core
python -m pip install -e ".[test]"    # + pytest, matplotlib (validation)
python -m pip install -e ".[numba]"   # optional JIT force kernel
```

Requires Python >= 3.10, NumPy, SciPy, h5py, PyYAML.

## Example run

```bash
python -m netsim.cli run configs/reference_star_N8_S.yaml --out run.h5
```

or from Python:

```python
from netsim import load_config
from netsim.simulate import simulate_config

cfg = load_config("configs/reference_star_N8_S.yaml")
result = simulate_config(cfg)
print(result.outcome, result.w_max, result.eta)
```

## Differences from the paper text

- **Time step uses the tangent wave speed.** The paper writes
  `dt = C min(ell_e/n_s)/c_L(eps_b)` with the *chord* speed `c_L`. For stability
  we use the maximum *tangent* wave speed
  `c_tan = max_eps sqrt((E0 + 3 b eps^2)/rho)` instead. For material S,
  `c_tan(eps_b) ~ 2480 m/s` vs `c_L(eps_b) ~ 1754 m/s`. The formula in the paper
  must be corrected to `c_tan`; we do not edit the paper. We also cap `dt` with a
  contact limit `C * 2/omega_c`, `omega_c = sqrt(k_eff/m_min)`,
  `k_eff = (3/2) k_c delta_ref^{1/2}`, `delta_ref = 0.01 r_d`. Both limits are
  logged along with the chosen step.
- **Two contact modes** (`frictionless` and `gripped`) are both implemented; see
  below. The reference off-centre solution (`ref/offc.py`) corresponds to
  `gripped`.
- **Optional sphere-segment contact** (point-segment distance) is available
  behind `contact.segment_contact`, off by default.
- **Kinematic forcing** mode (prescribed `v(t)` or `w(t)` at a node) replaces the
  drone for validation tests.
- **No damping by default.** Viscous damping is available as an option
  (`numerics.damping`), default 0.

## Contact modes

- `frictionless`: Hertz penalty `f_c = k_c delta^{3/2} n_hat` between the sphere
  and net nodes (`delta = r_d - |x_i - x_d| > 0`); equal/opposite force on the
  drone; no friction.
- `gripped`: at first contact the net node nearest the impact point is rigidly
  attached to the drone (its mass added to `M`); its segment forces are
  transmitted to the drone; all other nodes still interact by the penalty force.

### Penalty stiffness (`k_c`) and penetration guard

`k_c` can be set two ways (`contact.k_c_mode`):

- `absolute` (default): `k_c` is used verbatim. Works well for the soft
  material S, but the stiff material D (`E0 = 110 GPa`, ~55x stiffer) is badly
  under-resolved at `k_c = 1e7`: the penetration `delta` reaches `delta_ref` and
  the energy balance blows up (`energy_error ~ 2.6`).
- `relative`: `k_c` is chosen so the contact frequency at `delta_ref` equals
  `k_c_factor` times a segment's axial frequency. With
  `k_eff = (3/2) k_c sqrt(delta_ref)` and `omega = sqrt(k/m_node)`, the node mass
  cancels and `k_c = k_c_factor^2 (E0 A / l_s) / ((3/2) sqrt(delta_ref))`,
  evaluated at the stiffest (shortest) segment. For D this gives
  `k_c ~ 1.2e9` at `k_c_factor = 4`, keeping `delta_max/delta_ref ~ 0.1` and
  bringing `energy_error` down by ~2 orders of magnitude. See
  `validation/contact_d.py` for the full both-modes report.

A runtime **penetration guard** (`contact.penetration_guard`) tracks `delta_max`
every step: `abort` (default) raises `PenetrationError` if `delta > delta_ref`
(the contact is under-resolved); `reduce_dt` shrinks `dt` from the actual `delta`
with a `penetration_margin` safety factor; `off` only logs `delta_max`. The
chosen `k_c`, both CFL limits and `delta_max` are recorded on the trajectory.

## Star with rings (force-density prestress)

`star_with_rings(N, R, radii, eps_p, q_ratio=...)` builds a star of `N` radials
plus concentric rings whose nodes are *shared* at each radial/ring crossing (no
overlapping threads). The ring prestress is set by the force-density method: the
radials get a force density `q_radial` (so they sit at about `eps_p`) and the
rings `q_ring = q_ratio * q_radial`; the equilibrium geometry is solved from the
force densities with the outer ring anchored, and rest lengths are recovered from
the solved geometry. With nonzero `q_ratio` the inner ring radii come out of the
FDM solve (higher `q_ratio` pulls the rings inward). The FDM equilibrium residual
`L_q x` vanishes at every free node (checked in `tests/test_rings_fdm.py`).

## Algorithm 2: minimum mass

`netsim/mmin.py` finds the lightest net that still defeats the drone, by
bisection on the uniform cross-section scale `s` (`A_e = s A_hat_e`, so
`m_net(s) = s m_net(1)`). The passing scale is first bracketed by doubling from
the analytical guess `s0` with `m_net = E_kin/e_mat`, then bisected to a relative
tolerance `tol`. Two criteria are supported: **A** (arrested and no failure) and
**B** (arrested, all failures within `R_max` of the impact point, fewer than
`k_max` failures). The criterion is evaluated for every impact point in a
configurable list and the worst (largest-mass) case is returned, together with
the worst impact point, the number of failed segments and the broken-segment
map. Per-point bisections are parallelised with `multiprocessing` and made
resumable: each `(point, s)` evaluation is cached as a small HDF5 file and
skipped if it already exists.

Because criterion B is *not* monotone in `s` in general, run the coarse
`monotonicity_scan` first (10 values between `0.25 s_hi` and `s_hi`) to confirm
the pass/fail pattern; `validation/mmin_study.py` does this for one case per
material. From the CLI:

```
python -m netsim.cli mmin configs/mmin_star_N8_S.yaml --criterion A --procs 4
python -m netsim.cli mmin configs/mmin_star_N8_S.yaml --scan 10
```

## Dynamic overload (constant-force forcing, thread removal)

`netsim/overload.py` implements the `constant_force` forcing mode and the
`remove_thread(e, t_b, t_f)` API used by the dynamic-overload test (loss of a
radial under constant load). A flat N-star is loaded by a constant vertical hub
force (derived analytically to give a target static strain `eps0`), damped to
equilibrium, after which one radial is removed -- instantaneously (`t_f = 0`) or
with its tension ramped to zero over a finite release time `t_f`. The hub carries
a configurable point mass `m_hub` so the response is governed by one light mode.
Reported quantities: `eps_s/eps0` (static after removal), `eps_m/eps0` (dynamic
peak), the dynamic amplification `DAF = eps_m/eps_s`, and the effective exponent
`n_eff = d ln F / d ln w` at the operating point. See `validation/test_daf.py`.

## Energy balance

The `/energy` dataset stores six components per output frame:
`[KE_drone, KE_net, U_elastic, U_failure, U_contact, dE_capture]`. The last
column, `dE_capture = (1/2) (M m)/(M+m) |v_node - v_drone|^2`, is the energy
dissipated in the inelastic capture of the gripped node; it is booked once at
grip time (zero before, and zero in frictionless mode). Without it the gripped
reference run shows a residual `energy_error` of exactly this inelastic loss
(~3.3e-5 of `E_norm` for the S reference); including it the gripped no-failure
run conserves energy to well under 1e-3. The relative error is normalized by
`E_norm = E_kin,0 + U_prestress`.

## Parameters not fixed by the paper

The paper leaves `k_c`, `r_d`, `R_max`, `k_max` as "?". The shipped configs use
explicit working values marked `# TODO: fix before production runs`:

| Parameter | Working value |
|-----------|---------------|
| `r_d`     | 0.15 m |
| `k_c`     | 1e7 N/m^{3/2} |
| `R_max`   | 0.5 m |
| `k_max`   | 10 |

## HDF5 output

One file per run. Datasets and attributes follow the paper's table; see
`netsim/io_hdf5.py`. The full config, git commit and package versions are stored
in `/params`.

## Validation

Scripts in `validation/` reproduce the tables/figures of the paper's
"Validation of the simulator" section. Each script states which paper values it
produces:

| Script | Paper content |
|--------|---------------|
| `validation/material_law.py`     | material-law numbers (S/D) |
| `validation/test1_smith.py`      | Test 1, single thread (Smith), CSV + eps(X) figure |
| `validation/test2_hub.py`        | Test 2, quasi-static hub impact `F(w)`, `w_b/R`, `eta_A` |
| `validation/junction_linear_2d.py` | transverse-corrected 2D junction theory |
| `validation/test3_junction.py`   | Test 3, junction vs 1D and 2D theory + finite amplitude vs `ref/` |
| `validation/test4_convergence.py`| Test 4, convergence in `n_s` |
| `validation/offcentre.py`        | off-centre `eta_A(a)` vs `ref/offc.py` / `offc_free.py` |
| `validation/energy.py`           | energy conservation vs `C` |
| `validation/contact_d.py`        | material-D contact resolution (Item 2) |
| `validation/test_daf.py`         | dynamic overload / DAF (Item 5a) |
| `validation/mmin_study.py`       | Algorithm 2 minimum mass (Item 5b) |
| `validation/export_paper.py`     | export all `paper_results/` CSVs (Item 6) |

### Junction: transverse-corrected 2D theory (reference)

For N >= 3 the full 2D mass-spring model radiates transverse waves into the
off-axis threads at the hub, an effect absent from the purely longitudinal 1D
Riemann reference (`ref/riemann.junction`). The correct linear prediction keeps
the 1D form with `S_N` replaced by

```
S_N' = S_N + (Z_T/Z_L) * sum_{j>=1} sin^2(phi_j)      (= S_N + (Z_T/Z_L) N/2 for N>=3, = S_N for N=2)
T0/Tinc   = 2 S_N'/(1+S_N')
dT_j/Tinc = -2 cos(phi_j)/(1+S_N')
```

with Lagrangian speeds at the prestress `c_L = sqrt((E0+3 b eps_p^2)/rho)` and
`c_T = sqrt(sigma(eps_p)/rho)` (the variant **without** the `(1+eps_p)` factor;
the `with` variant is indistinguishable at the prestresses tested), and
`Z_T/Z_L = c_T/c_L`. The simulator matches this 2D theory within a few percent
for all N, and the 1D-numerical discrepancy scales as `Z_T/Z_L ~ sqrt(eps_p)`
(verified for material D). This 2D formula is the reference used in
`validation/test3_junction.py` and `tests/test_junction.py`.

**Measurement window.** `T0` and `dT_j` are read from the strain plateau 5
segments from the hub, averaged over `t in [tarr + 0.6 L/c, tarr + 1.5 L/c]`
(`tarr = L/c` the incident arrival time): after the junction reaches its steady
state, but before reflections from the far anchors return (~`2 L/c` after
arrival).

### Round-3 notes (checks before `--full`)

- **Criterion B.** Implemented as
  `arrested and (n_failed == 0 or (n_failed < k_max and R_d <= R_max))`.
  Zero-failure arrest therefore always passes B. A timeout after the drone has
  already reversed (`had_negative_vz` and `vz >= 0`) is counted as arrest.
  Because B is not monotone in `s`, Algorithm 2 for B is a scan (≥ 20 values)
  plus local bisection; both `s_min_first_pass` and `s_min_all_pass` are kept.
- **Gripped capture bookkeeping.** At grip the drone velocity is reset to the
  centre-of-mass velocity of `(M, m)` and `dE_cap = ½ μ |v_rel|²` is booked.
  Leaving `vd` unchanged while adding the node mass produced a C-independent
  energy floor of order `dE_cap` (~6.5e-4).
- **Smith longitudinal front (S).** Front speed is the slope of
  `X_front(t)` over ≥ 5 late times (not `X/t`). The residual ~2% plateau for S
  is the dispersive shock structure of the discrete chain (stiffening material),
  not a start-up offset.
- **Ring at R/2.** `NetConfig.fix_radii=True` (default) keeps ring nodes at the
  prescribed radii and sets different radial force densities on the inner/outer
  spans from ring-node equilibrium. Free FDM (`fix_radii=False`) still pulls the
  ring inward (~0.235 R at `q_ratio=0.5`).
- **Frictionless sensitivity.** At `a=R/2`, `N=8`, the gap between radials is
  `2π a/N ≈ 0.39 m > 2 r_d = 0.30 m`, so the frictionless drone can pass between
  radials (`w_max/R ≈ 2.15`, 0 failures). That is geometry, not a bug. Gripped is
  the baseline contact model.

### Round-4 notes (D contact, lateral drift)

- **Early perforation.** Besides `z < -2R`, the run stops as `perforated` when
  the drone is still descending and no intact segment is in contact or connected
  to the gripped node (point of no return). Failures up to that time are kept.
- **D contact default.** Arresting-case sweeps (`s = 1.1 sA_min`) show outcome,
  failure set, `w_max` and `η` stable to < 1% for `k_c_factor` ∈ {4,8,16}. Default
  for material D is now `k_c_factor=8`. Energy error on arresting D runs is ~1e-6.
- **Failure energy.** Booked at the interpolated crossing `eps = eps_b` (not the
  overshot `eps_new`); local dt shrinks near breaking. Single-thread work balance
  is then ~1e-3 relative (D at the threshold, S a few ×10⁻³).
- **Lateral drift.** `KinematicConfig.free_lateral` leaves the in-plane DOFs free
  (matches `ref/offc_free.py`). `dyn_runs.csv` / `tab_conv.csv` carry
  `drone_x_arrest`, `drone_y_arrest`; `fig_etaa.csv` carries `etaA_free_num`,
  `etaA_free_ref`, `px_fail`.
- **Paper CSV export.** `python -m validation.export_paper [--full] [--jobs N]`
  writes every file in `paper_results/SPEC.md`. Independent cases
  (`fig_etaa`, ring, `tab_mmin`, `dyn_runs`) use a process pool (`--jobs`
  defaults to all CPUs).

Run the pytest suite:

```bash
cd "Tema 2"
python -m pytest -q
```
