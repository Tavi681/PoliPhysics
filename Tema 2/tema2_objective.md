## Context

Implement an explicit mass–spring simulator for the impact of a drone on a prestressed planar net, with thread failure. The implementation follows Algorithm 1 of the paper `net_students_en.tex` (Section "Numerical method"). Write code, comments, docstrings, and log messages in English. Use SI units throughout.

Stack: Python ≥3.10, `numpy`, `scipy`, `h5py`, `numba` (optional, for the force loop), `pyyaml`, `pytest`, `matplotlib` (validation scripts only). Do not add other dependencies without asking.

Do not modify the scripts in `ref/` (`offc.py`, `riemann.py`, `sim.py`, `tables.py`). You may only read them, as reference values.

## Physical model (exactly as in the paper)

**Net.** The net is a planar graph \(G=(V,E)\) with anchored nodes \(\mathcal{A}\) on a rigid frame of radius \(R\), in the plane \(z=0\). Each thread \(e\) has:
- rest length \(\ell_e\);
- cross-section \(A_e\);
- density \(\rho\).

The net mass is \(m_{net}=\sum_e \rho A_e \ell_e\).

**Constitutive law.** The nominal tension is \(T(\epsilon)=A(E_0\epsilon+b\epsilon^3)\) for \(0<\epsilon<\epsilon_b\), and \(T=0\) for \(\epsilon\le 0\) (no compression, no bending stiffness). The strain-energy density is \(\Phi(\epsilon)=E_0\epsilon^2/2+b\epsilon^4/4\), and \(e_{mat}=\Phi(\epsilon_b)/\rho\).

The two model materials:

| Material | \(E_0\) (Pa) | \(b\) (Pa) | \(\epsilon_b\) | \(\rho\) (kg/m³) |
|---|---|---|---|---|
| S | 2e9 | 22.2e9 | 0.30 | 1300 |
| D | 110e9 | 0 | 0.032 | 975 |

Put them in a `MATERIALS` registry, with the option of adding materials from YAML.

**Prestress (force-density method).** Solve \(L_q^{ff}x_f=-L_q^{fa}x_a\), where \(L_q\) is the graph Laplacian weighted by \(q_e\). Then:
- \(l_e\) is the current thread length;
- \(T_e=q_e l_e\);
- \(\epsilon_e\) is obtained by inverting \(T(\epsilon_e)=T_e\) (Newton on the cubic), checking \(0\le\epsilon_e<\epsilon_b\);
- \(\ell_e=l_e/(1+\epsilon_e)\).

For the star with \(N\) radials, also provide a direct constructor with uniform prestress \(\epsilon_p\): hub at the origin, \(\ell=R/(1+\epsilon_p)\).

**Discretization.**
- Each thread is split into \(n_s\) equal segments along the straight line between its end nodes, with rest length \(\ell_e/n_s\). The initial state is thus exactly in equilibrium.
- Masses are lumped at the nodes: \(m_i=\sum_{e\ni i}\tfrac12\rho A_e\ell_e/n_s\).
- Anchored nodes are fixed.
- The force on segment \(ij\) is \(f_{ij}=T(\epsilon_{ij})\,(x_j-x_i)/\|x_j-x_i\|\), with \(\epsilon_{ij}=\|x_j-x_i\|/\ell_{ij}-1\).
- Keep the maps segment → parent thread \(e\), and segment node → (thread, position) or graph node.

**Drone.** The drone is a rigid sphere of mass \(M\) and radius \(r_d\). It starts with its centre at \((p_x,p_y,r_d)\), i.e. tangent to the net, with velocity \((0,0,-v_0)\). Gravity is off by default (flag `gravity: false`).

## Contact (two modes, both mandatory)

The analytical results in the paper depend on the contact model: whether the thread is gripped by the drone or slides over it. Implement two modes, selected by `contact.mode`:

1. **`frictionless`.** Penalty force \(f_c=k_c\,\delta^{3/2}\,\hat n\) between the sphere and the net nodes, where \(\delta=r_d-\|x_i-x_d\|>0\) and \(\hat n\) is the radial normal of the sphere. The equal and opposite force acts on the drone. No friction.
2. **`gripped`.** At first contact, the net node closest to the impact point \(p\) is rigidly attached to the drone: it follows the contact position, and its mass is added to \(M\). All other nodes interact with the drone through the penalty force, as in mode 1.
   - This is the model used in the reference solution for the off-centre impact (`ref/offc.py`): the drone grips the thread at a point.
   - It can be implemented as a constrained node. The resultant segment force on that node is transmitted to the drone.

**Discretization check.** Check and log that the segment-node spacing \(\ell_e/n_s\) is much smaller than \(r_d\). By default, warn if it exceeds \(r_d/4\).

Also add an optional sphere–segment contact mode (point–segment distance) behind a flag. Do not make it the default.

## Time integration (Algorithm 1)

Use explicit velocity Verlet with a fixed step. Order within one step:

1. \(x^{n+1}=x^n+v^n\Delta t+\tfrac12 a^n\Delta t^2\) for free nodes and the drone. Anchors stay fixed; in `gripped` mode the gripped node follows the drone.
2. Compute segment strains at \(x^{n+1}\).
3. Permanently remove all segments with \(\epsilon_{ij}\ge\epsilon_b\), even if several fail in the same step.
   - Append to the failure list: segment index, parent thread, time \(t\), and position of the segment midpoint.
   - Book the elastic energy of the failed segment, \(A\,(\ell_e/n_s)\,\Phi(\epsilon_{ij})\), as energy dissipated by failure.
4. Compute tensions (zero if the segment is slack), then contact forces, then \(a^{n+1}\).
5. \(v^{n+1}=v^n+\tfrac12(a^n+a^{n+1})\Delta t\).
6. Save an output frame every `dt_out`.
7. Check the stopping criteria (below).

**Time step.** The paper writes \(\Delta t=C\,\min(\ell_e/n_s)/c_L(\epsilon_b)\), \(C\le0.5\), where \(c_L\) is the chord speed \(\sqrt{T/(\mu_0\epsilon)}\). For stability, use instead the maximum tangent wave speed on \([0,\epsilon_b]\):

\[c_{tan}=\max_\epsilon\sqrt{(E_0+3b\epsilon^2)/\rho}\]

- For S, \(c_{tan}(\epsilon_b)\approx2480\) m/s, versus 1754 m/s for the chord speed.
- Take the minimum with a contact limit: \(\Delta t\le C\cdot 2/\omega_c\), where \(\omega_c=\sqrt{k_{eff}/m_{min}}\) and \(k_{eff}=\tfrac32 k_c\delta_{ref}^{1/2}\). Use a conservative estimate for \(\delta_{ref}\), e.g. 1% of \(r_d\).
- Log both limits and the chosen step.
- State in the README that the formula in the paper must be corrected to \(c_{tan}\). Do not edit the paper yourself.

**Stopping criteria.**
- `arrested`: the vertical velocity of the drone changes sign, i.e. \(v_z\) becomes \(\ge 0\) after having been negative. Record \(w_{max}=\) maximum drone displacement along \(-z\).
- `perforated`: the drone goes below \(z=-2R\).
- `timeout`: \(t\ge t_{end}\). This is not an error, but it must be flagged explicitly.

**Energy balance (mandatory, at every output frame).** Save:
- kinetic energy of the drone and kinetic energy of the net;
- elastic energy of the intact segments;
- energy dissipated by failures;
- contact energy, \(\tfrac25 k_c\delta^{5/2}\).

Report at the end the relative energy error, normalized by \(E_{kin,0}+U_{prestress}\). The integrator has no damping. Add viscous damping only as an option, default 0.

## Outputs (attributes of `/outcome`)

- `arrested` (bool), `outcome` (string), `w_max`;
- \(\eta\) = energy absorbed by the net (elastic + dissipated by failure, minus the initial prestress energy) divided by \(m_{net}e_{mat}\), at arrest;
- `n_failures`;
- \(R_d\) = maximum in-plane distance between the midpoint of a failed segment and \(p\);
- `cascade`: `n_failures >= k_max` or \(R_d>R_{max}\). \(R_{max}\) and \(k_{max}\) are configuration parameters, with no hidden defaults.

## HDF5 layout (one file per run, exactly as in the paper's table)

| Dataset / attribute | Shape | Content |
|---|---|---|
| `/params` (attributes) | — | material (E0, b, rho, eps_b), eps_p, R, M, r_d, v0, p, contact mode, n_s, C, dt, k_c, git commit |
| `/graph/nodes` | (n_v,2) | node positions of the prestressed net |
| `/graph/edges` | (n_e,2) | thread connectivity |
| `/graph/anchored` | (n_v) | anchored flag |
| `/graph/q`, `/graph/A` | (n_e) | force densities, cross-sections |
| `/t` | (n_t) | output times |
| `/x`, `/v` | (n_t, n_seg_nodes, 3) | positions and velocities of segment nodes |
| `/intact` | (n_t, n_seg) | intact mask per segment |
| `/drone` | (n_t, 6) | drone position and velocity |
| `/failures` | (n_f, 3) | segment index, parent thread, failure time |
| `/energy` | (n_t, 5) | energy-balance components |
| `/outcome` (attributes) | — | arrested, w_max, eta, R_d, cascade, energy_error |

Also add `/seg/edges` (n_seg,2), `/seg/parent` (n_seg), and `/seg/rest_length` (n_seg), so the fine graph can be reconstructed. Write datasets with `compression="gzip"` and chunking along the time axis.

## Code structure

```
netsim/
  materials.py      # law T, tangent, Phi, e_mat, inverse T -> eps, c_L, c_T, c_tan
  topology.py       # star(N,R,eps_p), star_with_rings(N,R,radii,...), from_arrays(nodes, edges, anchored, q, A)
  fdm.py            # force-density solve, rest lengths
  discretize.py     # subdivision, lumped masses, maps
  contact.py        # frictionless penalty, gripped constraint, optional segment contact
  integrator.py     # velocity Verlet loop (numba-jitted force kernel, numpy fallback)
  energy.py
  io_hdf5.py
  simulate.py       # simulate(config) -> Result, the Algorithm 1 entry point
  cli.py            # python -m netsim.cli run config.yaml
configs/            # reference_star_N8_S.yaml, etc.
validation/         # scripts that produce the tables of Sec. "Validation of the simulator"
tests/
```

Main function: `simulate(net, material, drone, numerics, contact, output) -> Result`. It takes dataclasses and optionally writes HDF5. Do not implement Algorithm 2 (bisection on the cross-section for the minimum mass) now. But prepare the API so that `simulate` can be called with all cross-sections scaled by a factor \(s\) (\(A_e=s\hat A_e\)) without rebuilding the topology.

Also support a **kinematic forcing** mode, instead of the drone, needed for validation: a given node receives a prescribed velocity \(v(t)\) or a prescribed displacement \(w(t)\).

## Validation (`pytest` tests + scripts in `validation/`)

Tolerances are indicative. If a test fails, do not change the physics or the tolerance to make it pass. Report the discrepancy and its likely cause.

1. **Material law (unit).**
   - For S: \(\sigma(\epsilon_b)=1.20\) GPa, \(e_{mat}=103.8\) kJ/kg, \(c_{L,0}=1240\) m/s, \(c_L(\epsilon_b)=1754\) m/s, and \(c_T(\epsilon_b)=\sqrt{T/(\mu_0(1+\epsilon))}=843\) m/s.
   - For D: \(\sigma_b=3.52\) GPa, \(e_{mat}=57.8\) kJ/kg, \(c_{L,0}=10622\) m/s.
2. **Test 1, single thread (Smith).** A long thread anchored at both ends, long enough that the fronts do not reach the anchors during the test. The middle node is given a constant prescribed transverse velocity \(v_0\) (100, 500, 1000 m/s). Compare with the analytical solution:
   - strain behind the longitudinal front, from \(v_0^2=\frac{T(\epsilon)}{\mu_0}\left[2\sqrt{\epsilon(1+\epsilon)}-\epsilon\right]\), solved numerically for \(\epsilon\);
   - position of the longitudinal front, \(c_Lt\), with \(c_L=\sqrt{T/(\mu_0\epsilon)}\);
   - position of the kink, \(c_Tt\), in the Lagrangian coordinate.

   Measure the fronts as the positions where \(\epsilon\) crosses half of the jump. Also check that the strain is continuous across the kink. Output: a CSV table with analytical and numerical values and relative errors, and a figure of \(\epsilon(X)\) at three times.
3. **Test 2, hub impact on the star** (quasi-static, slow prescribed displacement of the hub, N=8).
   - Compare \(F(w)=N\,T(\epsilon(w))\,w/\sqrt{R^2+w^2}\), with \(1+\epsilon(w)=(1+\epsilon_p)\sqrt{1+(w/R)^2}\).
   - Compare the deflection at failure with \(w_b/R\): S 0.831 / 0.770 / 0.650 and D 0.255 / 0.241 / 0.212, for \(\epsilon_p/\epsilon_b=0, 0.1, 0.3\).
   - Compare \(\eta_A=1-\Phi(\epsilon_p)/\Phi(\epsilon_b)\): S 1.000 / 0.993 / 0.937 and D 1.000 / 0.990 / 0.910.
   - The prescribed speed must be well below \(c_T\). Check rate independence with two speeds.
4. **Test 3, junction.** Star with N ∈ {2, 3, 4, 5, 6, 8, 12, 16} and long threads. A small tension pulse is sent along thread 0, through a short prescribed longitudinal velocity at the anchor of thread 0. Compare with the linear theory:
   - \(T_0/T_{inc}=2S_N/(1+S_N)\);
   - \(\Delta T_j/T_{inc}=-2\cos\varphi_j/(1+S_N)\);
   - \(S_N=1\) for N=2 and \(S_N=N/2-1\) for N≥3.

   The prestress must be large enough that same-side threads do not go slack. Then repeat at large amplitudes and compare with `ref/tables.py` (finite-amplitude junction, \(\epsilon_p=0.1\epsilon_b\)). Import the functions from `ref/`; do not copy them.
5. **Test 4, convergence** in \(n_s\) = 10, 20, 40, 80, on the reference case: star N=8, material S, M=1 kg, \(v_0\)=15 m/s, off-centre impact at \(a=R/2\), both contact modes. Report \(w_{max}/R\), number of failed segments, \(\eta\), and CPU time, with the relative difference from \(n_s=80\).
6. **Off-centre impact reference** (quasi-static, `gripped` mode, slow prescribed displacement of point \(p\) on thread 0, N=8, \(\epsilon_p=0\)).
   - Compare \(\eta_A(a)\) with `ref/offc.py`: at \(a/R=0.5\), \(\eta_A\approx0.068\) (S) and 0.069 (D); at \(a/R=0.05\), 0.176 (S) and 0.224 (D).
   - Also compare which segment fails first: inner for \(a/R\le0.1\), outer for \(a/R\ge0.2\).
   - Repeat in `frictionless` mode and report the differences without correcting them. In this mode radial 0 is expected to fail as a whole.
7. **Energy conservation.** On the reference case without failures (small M, small \(v_0\)), the relative energy error must be < 1e-3 for C=0.5 and must decrease with C.

## Quality requirements

- Vectorized or `numba` force loop. Target: the reference case N=8, \(n_s\)=40, up to arrest, in under one minute on a laptop.
- Determinism: no randomness. If any is introduced, it is controlled by a `seed` in the config.
- Every run stores the full config, the git commit, and the package versions in the HDF5 file.
- `README.md` describes:
  - installation;
  - an example run;
  - differences from the paper text: the time step with \(c_{tan}\), the two contact modes, and any other deviation;
  - which values from the paper's tables each script in `validation/` produces.
- Do not invent values for the parameters marked "?" in the paper (\(k_c\), \(r_d\), \(R_{max}\), \(k_{max}\)). Put them in the config with explicit working values, marked `# TODO: fix before production runs`.

At the end, give me a summary:
- which tests pass and which fail, with numbers;
- which implementation decisions you took that are not fixed by the paper.