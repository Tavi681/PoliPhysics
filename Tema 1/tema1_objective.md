You are implementing "Algorithm 1" of a physics paper EXACTLY as written: implicit discrete
elastic rods (DER) time stepping for a multi-thread charged "ballooning spider" in an air flow.
The code will be shown to journal reviewers, so it must map one-to-one onto the algorithm.
I will review and modify the code myself. Do not simplify the method, do not add physics.

## Traceability requirement (most important)
- Every step of Algorithm 1 (lines 1–23 below) must appear in integrator.py as a clearly
  delimited block starting with a comment "# Alg.1 line <k>: <text of the line>".
- Every force/energy term must cite its equation: "# Eq. (x)" using the numbering given below.
- Generate docs/algorithm_mapping.md: a table "Algorithm line | file:function | notes".
- No hidden heuristics. Any numerical choice not stated here must be a named config parameter
  with a TODO comment and listed in your final summary.

## Tech stack
Python 3.11, NumPy, SciPy (sparse, sparse.linalg), h5py, pytest, matplotlib (tests/plots only).
No autodiff: analytic gradients and Hessians, verified by finite differences in tests.
Layout:
  ballooning/params.py, geometry.py, frames.py, elastic.py, forces.py, fields.py,
  integrator.py, observables.py, io_hdf5.py, cli.py
  tests/, examples/, docs/algorithm_mapping.md

## Degrees of freedom (as in the paper)
Node 0 = spider. Thread j has N_t edges and nodes x_{(j-1)N_t+1} ... x_{jN_t}; its first edge
connects node 0 to x_{(j-1)N_t+1}. Total nodes n = N N_t + 1, edges N N_t.
Each edge k carries a twist angle theta^k.
xi = [x_0, x_1, ..., x_{N N_t}, theta^0, ..., theta^{N N_t - 1}] in R^{n_dof},
n_dof = 3(N N_t + 1) + N N_t.

## Elastic energy E_el = E_s + E_b + E_t (Bergou et al. 2008, Discrete Elastic Rods)
Per thread, reference edge length l0 = L/N_t, Voronoi length at an interior node i:
l_i = (|e0_{i-1}| + |e0_i|)/2 (undeformed).
- Stretching (Eq. S): E_s = 1/2 * Y A * sum_k (|e_k|/l0 - 1)^2 * l0, A = pi r^2.
- Bending (Eq. B): curvature binormal kb_i = 2 (e_{i-1} x e_i)/(|e_{i-1}||e_i| + e_{i-1}.e_i);
  material curvatures kappa_i = (kb_i . m2, -kb_i . m1) averaged over the two adjacent edges'
  material frames; E_b = sum_i (Y I / l_i) |kappa_i - kappa0_i|^2, I = pi r^4/4, kappa0 = 0.
- Twist (Eq. T): m_i = theta^i - theta^{i-1} + reference twist of the reference frames at node i;
  E_t = sum_i (G J / l_i) m_i^2, G = Y/(2(1+nu)), nu = 0.5 (config), J = pi r^4/2.
- Bending/twist stencils do not cross node 0 between different threads: each thread's first
  edge couples to node 0 only through stretching (free spider junction, as in the paper).
Provide analytic gradients and sparse Hessians for all three terms (standard DER formulas).

## Frames (frames.py)
- Reference frame (d1, d2) per edge, updated each Newton iteration by parallel transport in time
  from the previous converged step (time-parallel transport, as in Bergou 2010 / DisMech).
- Reference twist at each interior node computed from the transported frames.
- Material frame: m1 = cos(theta) d1 + sin(theta) d2, m2 = -sin(theta) d1 + cos(theta) d2.

## External forces f_ext (Eq. (16) of the paper: W + F_v + F_r + F_l), SI units
Defaults: m=1e-6 kg, r_s=1e-3, Q_s=3e-12 C, r=300e-9, L=0.5, rho_t=1200, Y=25e9,
mu=1.837e-5 Pa s, g=9.81, k_e=8.9875517923e9.
- F_v (Eq. (15)): at node k with Voronoi length dl_k and node tangent t_k
  (normalized average of adjacent edge tangents):
  F_v,k = -[eta_par t t^T + eta_perp (I - t t^T)] dl_k (v_k - u(x_k, t)),
  eta_par = 2 pi mu/(ln(L/r) - 1/2), eta_perp = 4 pi mu/(ln(L/r) + 1/2).
  Spider node: additional Stokes drag -6 pi mu r_s (v_0 - u(x_0, t)).
- F_l: q_k E(z_k) z_hat. Charge models: "tip" (q at the last node of each thread) or
  "uniform" (q dl_k / L on thread nodes); q_0 = Q_s.
- F_r (Eq. (6)): k_e sum_{i != k} q_i q_k (x_k - x_i)/|x_k - x_i|^3, analytic Jacobian
  (dense for "uniform").
- W: weight of lumped masses (m at node 0, rho_t A dl_k at thread nodes). Twist DOFs have no
  external force.
Field models E(z): "gorham" E0 exp(-alpha z) (E0=120, alpha=3e-4); "chamber"
E1 e^{-z/z1} + E2 e^{-z/z2} + Einf (E1=2.52e5, z1=1.51e-3, E2=5.07e4, z2=7.93e-3, Einf=7.41e3);
"constant". Air velocity: a callable u(x, t); implement ZeroFlow and UniformFlow now; leave a
documented interface for a KinematicSimulation class (to be implemented later).

## Mass matrix
Lumped, diagonal: m at node 0, rho_t A dl_k at thread nodes (3 entries each); twist DOFs get the
rotational inertia rho_t J l_k / 2 (standard DER choice; config parameter).

## Algorithm 1 (implement exactly these lines, in this order)
Require: parameters P, field E(z), air velocity u(x,t); initial state xi^0 (threads vertical,
spaced d0 = 100 um on a circle around the spider, pointing up), xi_dot^0 = 0; dt0, dt_max,
tolerance eps, max Newton iterations K, final time t_end.
1:  n <- 0, t <- 0, dt <- dt0, c <- 0            (c: consecutive successful steps)
2:  compute lumped mass matrix M
3:  while t < t_end do
4:    xi^(0) <- xi^n; k <- 0                    (Newton initial guess)
5:    repeat
6:      update reference frames by parallel transport; compute material frames from xi^(k)
7:      v^(k) <- (xi^(k) - xi^n)/dt
8:      f_ext <- W + F_v(xi^(k), v^(k), u) + F_r(xi^(k)) + F_l(xi^(k))
9:      assemble residual F (Eq. (17)) and Jacobian J (Eq. (18))
10:     solve J dxi = F; xi^(k+1) <- xi^(k) - dxi; k <- k + 1
11:   until ||F||_1 < eps or k = K
12:   if ||F||_1 >= eps then                      (Newton failed)
13:     dt <- dt/10; c <- 0; retry the step
14:   else
15:     xi^{n+1} <- xi^(k); xi_dot^{n+1} <- (xi^{n+1} - xi^n)/dt; t <- t + dt; n <- n + 1
16:     c <- c + 1; if c = 10 then dt <- min(10 dt, dt_max), c <- 0
17:     save (t, xi^n) at fixed output interval (HDF5)
18:     if |zdot_0^n - zdot_0^{n-1}| / |zdot_0^n| < delta over a window t_w then
19:       return trajectory, steady-state flag = true
20:     end if
21:   end if
22: end while
23: return trajectory, steady-state flag = false

Residual (Eq. (17)): F(xi) = M/dt (( xi - xi^n)/dt - xi_dot^n) + grad E_el(xi) - f_ext(xi).
Jacobian (Eq. (18)): J = M/dt^2 + Hess E_el - d f_ext/d xi. The drag term contributes C/dt with C
the block-diagonal RFT resistance matrix; as stated in the paper, the dependence of t_k on xi is
lagged (t_k taken from the current Newton iterate, not differentiated). Make "lag_tangent=True"
the default and document it.
Additional stopping rules used by Algorithm 2 (config, off by default): z_0 >= h -> "rise",
z_0 <= 0 -> "fall". They are checked at line 18 alongside the steady-state test and must be
labeled "# Alg.2 stopping rule".
Defaults: dt0 = 1e-4 s, dt_max = 1e-2 s, eps = 1e-10 (N, on ||F||_1), K = 20, delta = 1e-6,
t_w = 0.05 s, output interval 1e-3 s. If dt < 1e-9 s, abort with a clear error.
Linear solve: scipy.sparse.linalg.spsolve; dense solve only when the "uniform" Coulomb Jacobian
is active.

## Observables (observables.py)
V = zdot_0; R = mean horizontal distance of tips from the bundle axis; theta_L = tip angle from
the vertical; d_min = min distance between nodes of different threads (excluding the two nodes
nearest the spider); tension T = Y A (|e|/l0 - 1) per edge; invariant T^beta sin(theta).

## HDF5 output, one file per run
/params (attrs): all physical and numerical parameters, field model, charge model, z0, h,
git commit hash. /turb: reserved (sigma_w, ell, U_h, lambda, seed, k_n, a_n, b_n, omega_n).
/t (n_t,), /x (n_t, n, 3), /v (n_t, n, 3), /theta (n_t, N N_t), /edges (N N_t, 2),
/thread_id (n,) with -1 for the spider, /q_node (n,),
/outcome (attrs): steady/rise/fall/timeout, exit time, entangled flag, V, R, theta_L.

## Tests (pytest), all must pass
1. Finite-difference checks of every gradient/Hessian/Jacobian (relative error < 1e-5),
   including twist and bending with random twisted configurations.
2. Zero elastic force for a straight, untwisted, unstretched thread; parallel transport of a
   straight thread leaves the frames unchanged.
3. Terminal velocity: N=1, tip charge, constant E, still air, steady state
   U = (N q E - m g)/(N eta_par L + zeta_s), zeta_s = 6 pi mu r_s, within 1%.
4. Habchi & Jawed test 1: N=1, m=1 mg, L=0.5 m, "chamber" field; report the steady vertical
   velocity (target ≈ 8.5 cm/s) and the time to reach it (≈ 0.1 s) for the charge in the config.
5. Uniform-flow invariance: UniformFlow(0.5 z_hat): steady shape in the spider frame equals the
   still-air shape (max deviation < 1e-6 L) and V differs by exactly 0.5 m/s.
6. First integral: N=2, tip charge, still air: T^beta sin(theta) constant along each thread
   within 2% (excluding the first and last two edges).
7. Twist stays ~0 (|theta| < 1e-8 rad) for isotropic threads with free ends (sanity check).
8. Convergence table for N_t = 25, 50, 100, 200 (printed, no pass/fail).

## Deliverables
Code, tests, docs/algorithm_mapping.md, examples/run_single.py, examples/convergence.py.
Final summary: file list, how to run tests, runtime of test 4, every deviation or ambiguity.
