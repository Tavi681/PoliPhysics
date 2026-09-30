"""Figures for Sec. VII from results/*.csv. Run from 'Tema 1':  python plot_results.py
Writes results/figs/fig_test1.pdf, fig_collapse.pdf, fig_invariant.pdf, fig_shapes.pdf, fig_invariance.pdf."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
R = "results"; O = os.path.join(R, "figs"); os.makedirs(O, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 150})
mu, g, m, L, r, E, ke = 1.837e-5, 9.81, 1e-6, 0.5, 3e-7, 7410.0, 8987551792.3
zs = 6*np.pi*mu*1e-3; lr = np.log(L/r); ep = 2*np.pi*mu/(lr-.5); en = 4*np.pi*mu/(lr+.5); b = en/ep
SN = lambda N: sum(1/np.sin(np.pi*j/N) for j in range(1, N))
def opening(N, Fl, vt):                      # Eqs. (gtau), (opening)
    q = Fl*m*g/(N*E); U = vt*(N*q*E-m*g)/(mu*N*L); tau = (m*g+zs*U)/(N*q*E); Lam = ke*q/(E*L**2)
    gt = (tau**(1-b)-1)/((b-1)*(1-tau)); return (SN(N)*Lam*gt/4)**(1/3)

# Fig. test1
d = pd.read_csv(f"{R}/fig_test1.csv"); f, ax = plt.subplots(1, 2, figsize=(7, 2.6))
for c, ls in [("4a", "-"), ("4b", "--")]:
    x = d[d.case == c]; lab = "tip charge" if c == "4a" else "charge on spider"
    ax[0].plot(x.t, x.z0*100, ls, c="k", label=lab); ax[1].plot(x.t[x.t > 0], x.vz0[x.t > 0]*100, ls, c="k", label=lab)
ax[1].axhline(8.5, ls=":", c="gray"); ax[1].set_xscale("log")
ax[0].set(xlabel="$t$ (s)", ylabel="$z_0$ (cm)", title="(a)"); ax[1].set(xlabel="$t$ (s)", ylabel="$v_z$ (cm/s)", title="(b)")
ax[0].legend(frameon=False); f.tight_layout(); f.savefig(f"{O}/fig_test1.pdf")

# Fig. collapse (needs R_over_L column in fig_collapse.csv)
d = pd.read_csv(f"{R}/fig_collapse.csv")
if "R_over_L" in d:
    f, ax = plt.subplots(figsize=(3.4, 3.0))
    for N, mk in [(2, "o"), (4, "s"), (8, "^")]:
        x = d[d.N == N]; ax.loglog([opening(N, a, v) for a, v in zip(x.Fl_bar, x.vt_num)], x.R_over_L, mk, mfc="none", c="k", label=f"$N={N}$")
    t = np.array([0.08, 0.35]); ax.loglog(t, t, "-", c="gray")
    ax.set(xlabel=r"$[S_N\Lambda g(\tau)/4]^{1/3}$", ylabel=r"$(R/L)_\mathrm{num}$"); ax.legend(frameon=False)
    f.tight_layout(); f.savefig(f"{O}/fig_collapse.pdf")
else:
    print("fig_collapse.csv has no R_over_L column: ask Cursor to add R_over_L and theta_L")

# Fig. invariant
d = pd.read_csv(f"{R}/fig_invariant.csv"); f, ax = plt.subplots(figsize=(3.4, 2.4))
ax.plot(d.s_over_L, d.invariant_norm, "k-"); ax.axhline(1, ls=":", c="gray"); ax.set_ylim(0.999, 1.001)
ax.set(xlabel="$s/L$", ylabel=r"$T^\beta\sin\theta$ / tip value"); f.tight_layout(); f.savefig(f"{O}/fig_invariant.pdf")

# Fig. shapes: (a-c) side view x-z in the spider frame, (d) top view of the tips
d = pd.read_csv(f"{R}/fig_shapes.csv"); f, ax = plt.subplots(1, 4, figsize=(7, 2.2))
lim = 1.1*np.abs(d[["x", "y"]]).values.max()
for i, N in enumerate([2, 4, 8]):
    for j, x in d[d.N == N].groupby("thread"):
        ax[i].plot(x.x, x.z, "k-", lw=0.8); tip = x.iloc[-1]
        ax[3].plot(tip.x, tip.y, ["o", "s", "^"][i], mfc="none", c="k", label=f"$N={N}$" if j == 0 else None)
    ax[i].set(xlim=(-lim, lim), xlabel="$x$ (m)", title=f"({'abc'[i]}) $N={N}$"); ax[i].set_ylabel("$z$ (m)" if i == 0 else "")
ax[3].set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xlabel="$x$ (m)", ylabel="$y$ (m)", title="(d) tips")
ax[3].legend(frameon=False, fontsize=6, loc="center")
f.tight_layout(); f.savefig(f"{O}/fig_shapes.pdf")

# Fig. invariance
d = pd.read_csv(f"{R}/fig_invariance.csv"); t = pd.read_csv(f"{R}/fig_invariance_t.csv")
f, ax = plt.subplots(1, 2, figsize=(7, 2.6))
for w, ls in [(0.0, "-"), (0.5, "--")]:
    for j, x in d[d.w == w].groupby("thread"): ax[0].plot(x.x_rel, x.z_rel, ls, c="k", lw=0.8)
    y = t[t.w == w]; ax[1].plot(y.t, y.z0, ls, c="k", label=f"$w={w}$ m/s")
ax[0].set(xlabel=r"$x-x_0$ (m)", ylabel=r"$z-z_0$ (m)", title="(a)"); ax[1].set(xlabel="$t$ (s)", ylabel="$z_0$ (m)", title="(b)")
ax[1].legend(frameon=False); f.tight_layout(); f.savefig(f"{O}/fig_invariance.pdf")
print("done:", os.listdir(O))
