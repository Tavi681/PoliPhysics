Stage A: turbulence generator, cheap studies, pilot. Keep the Alg.1 traceability comments.
New code must cite the paper's LaTeX labels (e.g. eq:ks, eq:vk, alg:sweep).

## 0. Constraints (read first)
- Commit the current state before starting (message: "Before Stage A").
- Additive only. Do not change existing defaults, public function signatures, or the behaviour of
  ZeroFlow/UniformFlow runs. New features are opt-in via config.
- The only allowed change to existing code is the du/dx term in the F_v Jacobian (item 1). It must be
  exactly zero for ZeroFlow and UniformFlow (skip the computation for these flows).
- Replace the KinematicSimulation stub in fields.py; keep its public interface u(x, t).
- Regression check after Stage A: run the full pytest suite and scripts/produce_results.py. All
  results/*.csv from the previous run must be identical (max relative difference < 1e-12, CPU and wall
  times excluded). Write the comparison to results/regression.txt. Keep the old CSVs as
  results/pre_stageA/ for this comparison.

## 1. KinematicSimulation (fields.py), paper Eq. (eq:ks) with spectrum Eq. (eq:vk)
u(x,t) = U_h x_hat + sum_n [a_n cos(k_n . x' + omega_n t) + b_n sin(k_n . x' + omega_n t)],
x' = x - U_h t x_hat.
- Wavenumbers: N_k geometric shells from k_min = 0.1/ell to k_max = 2 pi/(5 dl), dl = L/N_t;
  dk_n = shell width. Report the fraction of (3/2) sigma^2 captured by the discrete modes; it must be
  >= 0.98, otherwise lower k_min and report the value used.
- Directions k_hat_n uniform on the sphere. a_n, b_n perpendicular to k_n, random orientation in the
  plane normal to k_n, |a_n| = |b_n| = sqrt(2 E(k_n) dk_n), so that
  (1/2) <|u - U_h x_hat|^2> = sum_n E(k_n) dk_n = (3/2) sigma^2.
- von Karman: E(k) = C sigma^2 ell (k ell)^4 / (1 + (k ell)^2)^(17/6), with C fixed numerically so that
  int_0^inf E dk = (3/2) sigma^2.
- omega_n = lambda sqrt(k_n^3 E(k_n)), lambda = 0.5.
- All randomness from one seed (numpy.random.Generator). Store seed, k_n, a_n, b_n, omega_n, sigma, ell,
  U_h, lambda, N_k in /turb of the HDF5 output.
- Vectorized evaluation for all nodes at once (no Python loop over nodes or modes in the hot path).
- Also provide grad_u(x, t) (3x3 per node), analytic.
- Jacobian: F_v,k = -D_k (v_k - u(x_k,t)) with D_k the RFT resistance tensor times dl_k. Add the term
  dF_v,k/dx_k = + D_k grad_u(x_k,t) to d f_ext/d xi (tangent dependence of D_k stays lagged, as now).
- Tests:
  a) divergence-free: |div u| < 1e-10 * max|k| * max|u| at random points;
  b) statistics over 1e5 random points and 200 seeds: std of each velocity component within 3% of sigma,
     mean within 3% of sigma/10;
  c) reproducibility: same seed -> identical field;
  d) finite-difference check of grad_u and of the new Jacobian term (relative error < 1e-5).
- Validation outputs (sigma = 0.25 m/s, ell = 1 m, U_h = 0, for N_k = 100 and 200):
  results/fig_ksvalid.csv: N_k, k, E_target, E_measured
    (1D longitudinal spectrum of w along z from a line of 4096 points spanning 20 ell, averaged over
     50 seeds, compared with the 1D spectrum implied by E(k) for isotropic turbulence);
  results/fig_ksvalid_pdf.csv: N_k, w_bin_center, pdf, gaussian (Gaussian with std sigma).

## 2. Clamped release (for Algorithm alg:sweep)
Option release_mode = "clamped": hold node 0 fixed (Dirichlet on its 3 DOFs) at z0 in still air until
the steady-state test passes, then release at t = 0 with the turbulent flow switched on. The thread
shape at release is the clamped steady shape. Label the block "# Alg.2 line 4". Default: off.

## 3. Table w_c (paper table tab:wc)
N = 1, 2, 4, 8; m = 0.1, 1, 10 mg; q = 0 and q = 0.6 nC per thread; tip charge; constant
E = 7.41e3 V/m; still air; Q_s = 0; N_t = 100. At steady state, w_c = -V.
Analytic: w_c = w_s (1 - Fbar_l), with w_s = m g/(N eta_par L + zeta_s) and Fbar_l = N q E/(m g).
Output results/tab_wc.csv: N, m_mg, q_nC, wc_numeric, wc_analytic, rel_err_pct.
(For q = 0 the bundle does not stay vertical by itself: report the steady falling shape as it is and
flag it in the log if the threads fold below the spider.)

## 4. Lateral relaxation (from results/equal_t_cache, transient runs with v0 = 0)
Compute for both runs R(t) (mean horizontal tip distance from the bundle axis), and fit exponential decay
times to |R_w(t) - R_0(t)| and to the max shape difference over t >= 0.5 s.
Output results/relax.csv: t, dR_over_L, dshape_over_L, and write both decay times to meta.json.

## 5. Pilot cost (Algorithm alg:sweep, not the production sweep)
N = 1, 2, 4, 8; Fbar_l = 1; sigma_w = 0.25 m/s; ell = 1 m; U_h = 1 m/s; N_k = 100; z0 = 0.5 m;
h = 2 m; t_end = 60 s; 3 seeds each; clamped release; adaptive dt; default eps and K.
Stopping: z_0 >= h (rise), z_0 <= 0 (fall), t >= t_end (timeout).
Report per run: N, seed, outcome, exit_time, simulated time, wall time, wall/simulated ratio, mean dt,
min dt, number of Newton failures, max |u| seen by the nodes.
Output results/pilot.csv. Run in parallel on the performance cores. Do NOT start the production sweep.

## 6. HDF5 for ML (prepare only)
Option to write trajectories in float32 at output_dt = 0.05 s, with /turb as in item 1 and /outcome.
Report the file size per simulated second for each N from the pilot runs (in meta.json).

## Final summary
List of changed and new files, test results, results/regression.txt verdict, captured-energy fraction,
pilot table, and every choice you made that is not specified here.
