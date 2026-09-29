# Algorithm 1 traceability map

This table maps every line of Algorithm 1 (and every force/energy equation) in
`tema1_objective.md` onto the code. Line numbers refer to the algorithm listing
in the objective. In `integrator.py` each line appears as a delimited block
starting with a comment `# Alg.1 line <k>: <text>`.

## Algorithm 1 lines

| Algorithm line | file:function | notes |
|---|---|---|
| Require (P, E(z), u(x,t), xi^0, xi_dot^0, dt0, dt_max, eps, K, t_end) | `params.py:Params`, `geometry.py:initial_state`, `fields.py:make_field`/`make_flow` | parameters, initial vertical threads on a circle of radius `d0`, zero velocity |
| 1: n<-0, t<-0, dt<-dt0, c<-0 | `integrator.py:simulate` | initialization block |
| 2: compute lumped mass matrix M | `integrator.py:build_mass_matrix` | diagonal; twist inertia `rho_t J l0/2` |
| 3: while t < t_end | `integrator.py:simulate` | main time loop |
| 4: xi^(0)<-xi^n; k<-0 | `integrator.py:simulate` | Newton initial guess |
| 5: repeat | `integrator.py:simulate` | Newton loop |
| 6: update reference frames (parallel transport); material frames | `integrator.py:_residual_and_jacobian` -> `frames.py:FrameState.evaluate` | time-parallel transport + material directors |
| 7: v^(k) <- (xi^(k)-xi^n)/dt | `integrator.py:_residual_and_jacobian` | nodal velocities |
| 8: f_ext <- W + F_v + F_r + F_l | `integrator.py:_residual_and_jacobian` -> `forces.py:assemble_fext` | Eq. (eq:fext) |
| 9: assemble residual F and Jacobian J | `integrator.py:_residual_and_jacobian` | Eq. (eq:residual), Eq. (eq:jacobian). |
| 10: solve J dxi = F; xi^(k+1)<-xi^(k)-dxi; k<-k+1 | `integrator.py:simulate` / `_solve` | `spsolve` (dense for uniform Coulomb) |
| 11: until ||F||_1 < eps or k = K | `integrator.py:simulate` | `||F||_1 = sum(abs(F))` |
| 12: if ||F||_1 >= eps (Newton failed) | `integrator.py:simulate` | convergence check |
| 13: dt <- dt/10; c<-0; retry | `integrator.py:simulate` | aborts if `dt < dt_min` |
| 14: else | `integrator.py:simulate` | accept step |
| 15: xi^{n+1}<-xi^(k); xi_dot^{n+1}<-(xi^{n+1}-xi^n)/dt; t<-t+dt; n<-n+1 | `integrator.py:simulate` | also commits the reference frame |
| 16: c<-c+1; if c=10 then dt<-min(10 dt, dt_max), c<-0 | `integrator.py:simulate` | step-size growth |
| 17: save (t, xi^n) at fixed output interval | `integrator.py:simulate` -> `io_hdf5.py:write_trajectory` | output every `output_dt` |
| 18: if steady-state test over window t_w | `integrator.py:simulate` | `|zdot_0^n - zdot_0^{n-1}|/|zdot_0^n| < delta` held for `t_w` |
| 19: return trajectory, steady-state flag = true | `integrator.py:simulate` | outcome status `steady` |
| 20-21: end if / end if | `integrator.py:simulate` | |
| 22: end while | `integrator.py:simulate` | |
| 23: return trajectory, steady-state flag = false | `integrator.py:simulate` | outcome status `timeout` |
| Alg.2 stopping rule: z_0>=h -> rise, z_0<=0 -> fall | `integrator.py:simulate` | off by default (`use_alg2_stopping`) |

## Equations

| Equation | file:function | notes |
|---|---|---|
| Eq. (eq:Es) stretching E_s | `elastic.py:stretch_energy_grad_hess` | `1/2 Y A sum (|e|/l0-1)^2 l0` |
| Eq. (eq:Eb) bending E_b | `elastic.py:bending_twist_grad_hess`, `_kappa_grad_hess` | curvature binormal, material curvatures, `kappa0=0` |
| Eq. (eq:Et) twist E_t | `elastic.py:bending_twist_grad_hess`, `_twist_grad_hess` | `m_i = theta^i - theta^{i-1} + ref twist` |
| Eq. (eq:coulomb) Coulomb F_r | `forces.py:_add_coulomb_force`, `external_position_jacobian` | Default `coulomb_include_spider=False`: sum over `i != 0`, `i != k` (spider is never a source), as in Habchi & Jawed. `coulomb_include_spider=True` sums over every `i != k`. Analytic Jacobian; dense solve for `uniform`. |
| Eq. (eq:Fvk) viscous F_v (RFT) | `forces.py:assemble_fext`, `resistance_matrix` | lagged node tangent; Stokes drag on spider |
| Eq. (eq:fext) f_ext = W+F_v+F_r+F_l | `forces.py:assemble_fext` | full external force vector |
| Eq. (eq:residual) residual F | `integrator.py:_residual_and_jacobian` | `M/dt((xi-xi^n)/dt - xi_dot^n) + grad E_el - f_ext` |
| Eq. (eq:jacobian) Jacobian J | `integrator.py:_residual_and_jacobian` | `M/dt^2 + Hess E_el + C/dt - d(F_l+F_r)/dxi` |
| Frames (parallel transport, reference twist, material frame) | `frames.py` | Bergou 2008/2010 conventions |
| Fields E(z): gorham/chamber/constant | `fields.py:GorhamField/ChamberField/ConstantField` | analytic dE/dz for Eq. (eq:jacobian) |
| Entanglement | `integrator.py:_entangled` | `d_min < entangle_contact_factor * r`, default factor 2 (threads in contact) |
| Air velocity u(x,t): ZeroFlow/UniformFlow | `fields.py` | `KinematicSimulation` reserved (stub) |

## Observables

| Quantity | file:function |
|---|---|
| V = zdot_0 | `observables.py:spider_vertical_velocity` |
| R (tip radius) | `observables.py:tip_radius` |
| theta_L (tip angle) | `observables.py:tip_angle` |
| d_min | `observables.py:min_interthread_distance` |
| tension T | `observables.py:edge_tensions` |
| invariant T^beta sin(theta) | `observables.py:first_integral` (default `beta = eta_perp/eta_par`) |

## Validation and production studies

All studies are defined in `ballooning/studies.py` and run by
`scripts/produce_results.py`, which writes `results/*.csv`, `results/meta.json`
and `results/log.txt`.

| Study | Setup | Pass/fail |
|---|---|---|
| Test 3 (`tab_valid`) | N=1, tip, constant E=8 kV/m, Q_s=0 and Q_s=3 pC; `U=(NqE+Q_sE-mg)/(N eta_par L+zeta_s)` | < 1% |
| Test 4a (`tab_valid`, `fig_test1`) | N=1, chamber field, m=0.9 mg, Q_s=0, spider Stokes drag on; `q=(m g+U_target(eta_par L+zeta_s))/E_inf`, `U_target=0.085` | `|V-0.085|/0.085 < 2%`; t_95 reported against `3 t_s`, `t_s=m/(eta_par L+zeta_s)` |
| Test 4b (`tab_valid`, `fig_test1`) | as 4a, the same q on the spider, no thread charge | report only |
| Test 5 / paper test 2 (`tab_valid`, `tab_conv`) | N=2, tip, constant E=7.41 kV/m, m=1 mg, Q_s=3 pC, `Fbar_l=2` | `|vbar_t-2|<0.25` |
| `tab_vt` | N=1, tip, constant E=7.41 kV/m, Q_s=0, `Fbar_l=2`, L=0.1/0.5/1 | report |
| `fig_collapse`, `tab_steady`, `fig_shapes`, `fig_invariant` | N=1,2,4,8, tip, constant E=7.41 kV/m, m=1 mg, L=0.5, Q_s=0, `q=Fbar_l m g/(N E)` | report |
| `fig_invariance` | N=4, `Fbar_l=2`, uniform flow w=0 and 0.5 m/s | report |
