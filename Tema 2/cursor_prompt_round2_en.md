# Cursor prompt, round 2: corrections and extensions to `netsim`

Continue working on `netsim` (Tema 2/). Same rules as before: do not edit the `.tex` files or `ref/`; do not tune physics or tolerances to make a test pass; report numbers. Work through the items in order and run `pytest` after each one.

## 1. Junction test: compare with the transverse-corrected theory

The deviation in Test 3 for N ≥ 3 must be checked quantitatively, not only explained. For a massless hub with small displacement, the off-axis threads resist the hub motion both longitudinally (impedance \(Z_L=\mu_0 c_L\)) and transversely (impedance \(Z_T=\mu_0 c_T\), both evaluated at the prestress \(\epsilon_p\)). The linear result keeps the same form as the 1D formula with \(S_N\) replaced by

\[
S_N' = S_N + \frac{Z_T}{Z_L}\sum_{j=1}^{N-1}\sin^2\varphi_j = S_N + \frac{Z_T}{Z_L}\,\frac{N}{2},\qquad
\frac{T_0}{T_{inc}}=\frac{2S_N'}{1+S_N'},\qquad
\frac{\Delta T_j}{T_{inc}}=-\frac{2\cos\varphi_j}{1+S_N'} .
\]

Tasks:
- Add `junction_linear_2d(material, N, eps_p)` to `validation/`, using the Lagrangian definitions \(c_L=\sqrt{T'(\epsilon_p)/\rho}\) and \(c_T=\sqrt{T(\epsilon_p)/(\mu_0(1+\epsilon_p))}\). Also try the variant without the \((1+\epsilon_p)\) factor, and report which one the simulation matches.
- Rerun Test 3 for N ∈ {2, 3, 4, 5, 6, 8, 12, 16}, materials S and D, \(\epsilon_p/\epsilon_b\) ∈ {0.02, 0.1, 0.3}, small pulse amplitude. Compare with both the 1D formula and the 2D formula.
- Scaling check: the 1D–numerical discrepancy must scale as \(Z_T/Z_L\propto\sqrt{\epsilon_p}\) (for D). Report it.
- Measure \(T_0\) and \(\Delta T_j\) on the plateau after the incident front has reached the hub and before any reflection from the anchors or transverse wave from the far end returns. Document the time window.
- If the 2D formula agrees within a few percent, write it in the README as the reference. If it does not, report the residual and do not tune anything.

## 2. Material D dynamic instability

Diagnose before fixing. Report the maximum penetration \(\delta_{max}\) versus \(\delta_{ref}=0.01\,r_d\) for the failing D case.
- Add a runtime guard: if \(\delta>\delta_{ref}\) at any step, either recompute \(\Delta t\) from the actual \(\delta\) (with a margin) or abort with a clear error. Make the behaviour configurable; the default is abort.
- Make \(k_c\) configurable relative to the axial stiffness of a segment: `k_c_mode: relative`, \(k_c\) chosen so that the contact frequency at \(\delta_{ref}\) equals a given multiple of the axial frequency of a segment, \(\sqrt{(EA/\ell_s)/m_{node}}\). Keep `absolute` as an option.
- Run the D reference case (N=8, M=1 kg, \(v_0\)=15 m/s, \(a=R/2\)) in both contact modes and report the energy error, \(w_{max}\), failures, and CPU time. Target: energy error < 1e-3.

## 3. Energy bookkeeping in `gripped` mode

Book the capture loss explicitly: \(\Delta E_{cap}=\tfrac12\frac{M m}{M+m}v_{rel}^2\), with \(m\) the captured node mass and \(v_{rel}\) the relative normal speed at capture. Add it as a sixth column of `/energy` and include it in the balance. Then add the gripped reference case (S, no failures) to the energy-conservation test with the same tolerance of 1e-3.

## 4. Test 1 and Test 4 details

- Test 1: report the front errors separately (longitudinal front, kink) for each material and \(v_0\), at \(n_s\) giving 200, 400, 800 segments over the resolved length. The error must decrease with resolution; if it does not, change the front detection (e.g. fit an error function to the profile), not the physics.
- Test 4: also report the number and positions of failed segments, \(\eta\), and \(R_d\) versus \(n_s\), for both contact modes. Convergence of \(w_{max}\) alone is not enough.

## 5. New features needed for the paper

**5a. Dynamic overload test (loss of a radial under constant load).** Add a forcing mode `constant_force`: a constant vertical force \(F\) applied to the hub of an N-star, with viscous damping used only to reach static equilibrium, then damping switched off. At \(t_b\) remove all segments of radial 0 at once (`remove_thread(e, t_b)` API) and record the maximum strain \(\epsilon_m\) in the remaining radials. Choose \(F\) so that the static strain before removal is \(\epsilon_0\) with two regimes: linear (material D) and cubic-dominated (material S at \(\epsilon_0\approx0.8\epsilon_b\)). Report \(\epsilon_s/\epsilon_0\) (static after removal, with damping) and \(\epsilon_m/\epsilon_0\) (dynamic) for N ∈ {4, 8, 16}, and the effective exponent \(n_{eff}=d\ln F/d\ln w\) measured at the operating point. Also implement a finite release time: the removed thread's tension decays linearly to zero over \(t_f\), for \(t_f/T_n\) ∈ {0, 0.25, 0.5, 1, 2, 4}, with \(T_n\) the period of the hub mode. For this test the hub carries a point mass \(m_{hub}\) (configurable), because the dynamic effect needs a light node, not a drone-held node.

**5b. Algorithm 2 (minimum mass).** Implement it now:
- bisection on the uniform cross-section scale \(s\), with \(A_e=s\hat A_e\), tolerance \(\delta\) (config);
- first find \(s_{hi}\) by doubling from an analytical guess \(s_0\) such that \(m_{net}=E_{kin}/e_{mat}\);
- the criterion is evaluated for every impact point in a list \(\{p_k\}\) (config), and the worst case is returned;
- criterion A: arrested and no failure; criterion B: arrested, all failures within \(R_{max}\) of \(p\), and fewer than \(k_{max}\) failures;
- return \(m^A_{min}\) or \(m^B_{min}\), the worst impact point, the number of failed segments, and the broken-segment map at the worst point;
- parallelize over impact points with `multiprocessing`; make runs resumable (skip existing HDF5 files).

Important: criterion B is not monotone in \(s\) in general (a heavier net can fail differently). Before trusting the bisection, run a coarse scan over \(s\) (10 values between \(0.25\,s_{hi}\) and \(s_{hi}\)) for one case per material and report whether the pass/fail pattern is monotone.

**5c. Star with rings.** Check that `star_with_rings(N, R, radii)` creates ring nodes at the radial crossings (shared nodes, not overlapping threads), with ring prestress set by the force-density method for given \(q_{ring}/q_{radial}\). Add a unit test on the FDM equilibrium residual.

## 6. Output for the paper

Add `validation/export_paper.py`, which reads the results and writes one CSV per table and figure into `paper_results/`, with exactly the column names listed in `paper_results/SPEC.md` (I will provide it). Each CSV starts with a comment header `# commit=<hash> date=<ISO> config=<file>`.

At the end, give me a summary with numbers for items 1–5, plus any decision you took that is not fixed by this prompt.
