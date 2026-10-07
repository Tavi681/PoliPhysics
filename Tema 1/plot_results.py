"""Publication figures for Sec. VII (still-air results and turbulence validation).

Run from 'Tema 1':   python plot_results.py
Reads results/*.csv, writes results/figs/<name>.pdf (for the paper, vector) and <name>.png (preview, 300 dpi).
Figures whose input CSV is missing are skipped with a message.

Style: figures4papers house style (sans-serif, top/right spines off, frameless legends,
semantic palette: blue = simulation, red = theory/reference, neutral = guides).
Figures are drawn at ~1.5x print size and scaled down in LaTeX (width = \\linewidth or 0.5\\linewidth).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
O = os.path.join(R, "figs")
os.makedirs(O, exist_ok=True)

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B", "red_strong": "#B64342", "red_2": "#E9A6A1",
    "neutral": "#CFCECE", "dark": "#4D4D4D", "teal": "#42949E", "violet": "#9A4D8E",
}
N_STYLE = {1: (PALETTE["dark"], "D"), 2: (PALETTE["blue_main"], "o"),
           4: (PALETTE["teal"], "s"), 8: (PALETTE["violet"], "^"),
           3: (PALETTE["red_strong"], "v"), 6: (PALETTE["green_3"], "P")}

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "font.size": 15,
    "axes.labelsize": 15, "axes.titlesize": 15, "legend.fontsize": 13,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 1.8, "xtick.major.width": 1.5, "ytick.major.width": 1.5,
    "xtick.minor.width": 1.0, "ytick.minor.width": 1.0,
    "lines.linewidth": 2.2, "lines.markersize": 8,
    "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.formatter.useoffset": False,
})

# physical constants of the still-air runs (Table VIII)
mu, g, m, L, r, E, ke = 1.837e-5, 9.81, 1e-6, 0.5, 3e-7, 7410.0, 8987551792.3
zs = 6 * np.pi * mu * 1e-3
lr = np.log(L / r)
ep = 2 * np.pi * mu / (lr - 0.5)
en = 4 * np.pi * mu / (lr + 0.5)
beta = en / ep


def S_N(N):
    return sum(1 / np.sin(np.pi * j / N) for j in range(1, N))


def opening_prediction(N, Fl, vt):
    """Eqs. (eq:gtau), (eq:opening); tau from the simulated U."""
    q = Fl * m * g / (N * E)
    U = vt * (N * q * E - m * g) / (mu * N * L)
    tau = (m * g + zs * U) / (N * q * E)
    Lam = ke * q / (E * L**2)
    gt = (tau**(1 - beta) - 1) / ((beta - 1) * (1 - tau))
    return (S_N(N) * Lam * gt / 4) ** (1 / 3)


def panel_label(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    fig.tight_layout(pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(O, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", name)


def have(*files):
    miss = [f for f in files if not os.path.exists(os.path.join(R, f))]
    if miss:
        print("skip:", ", ".join(miss), "missing")
    return not miss


# ---------------------------------------------------------------- Fig. test1
if have("fig_test1.csv"):
    d = pd.read_csv(f"{R}/fig_test1.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    cases = [("4a", "charge at tip", PALETTE["blue_main"], "-"),
             ("4b", "charge on spider", PALETTE["red_strong"], "--")]
    for c, lab, col, ls in cases:
        x = d[d.case == c]
        ax[0].plot(x.t, x.z0 * 100, ls, color=col, label=lab)
        y = x[x.t > 0]
        ax[1].plot(y.t, y.vz0 * 100, ls, color=col, label=lab)
    ax[1].axhline(8.5, ls=":", color=PALETTE["dark"], lw=1.5)
    ax[1].text(0.97, 0.27, "8.5 cm/s (Habchi & Jawed)", transform=ax[1].transAxes, ha="right",
               color=PALETTE["dark"], fontsize=11)
    ax[1].set_xscale("log")
    ax[0].set(xlabel="$t$ (s)", ylabel="$z_0$ (cm)")
    ax[1].set(xlabel="$t$ (s)", ylabel="$v_z$ (cm/s)")
    ax[0].legend(loc="upper left")
    panel_label(ax[0], "(a)"); panel_label(ax[1], "(b)")
    save(fig, "fig_test1")

# ------------------------------------------------------------- Fig. collapse
if have("fig_collapse.csv"):
    d = pd.read_csv(f"{R}/fig_collapse.csv")
    if "R_over_L" not in d:
        print("skip fig_collapse: no R_over_L column")
    else:
        d = d[d.N > 1].copy()
        d["pred"] = [opening_prediction(n, f, v) for n, f, v in zip(d.N, d.Fl_bar, d.vt_num)]
        fig, ax = plt.subplots(2, 1, figsize=(5.2, 6.2), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1.2]})
        lo, hi = 0.8 * d.pred.min(), 1.15 * d.pred.max()
        t = np.array([lo, hi])
        ax[0].plot(t, t, "-", color=PALETTE["neutral"], lw=2.5, zorder=0, label="Eq. (opening)")
        for N in sorted(d.N.unique()):
            x = d[d.N == N]; col, mk = N_STYLE[N]
            ax[0].plot(x.pred, x.R_over_L, mk, color=col, mfc="white", mew=1.8, label=f"$N={N}$")
            ax[1].plot(x.pred, 100 * (x.R_over_L / x.pred - 1), mk, color=col, mfc="white", mew=1.8)
        ax[1].axhline(0, color=PALETTE["neutral"], lw=2.5, zorder=0)
        dev = 100 * (d.R_over_L / d.pred - 1)
        pad = 0.15 * (dev.max() - dev.min() + 1)
        ax[1].set_ylim(min(dev.min(), 0) - pad, max(dev.max(), 0) + pad)
        ax[0].set(xlim=(lo, hi), ylim=(lo, hi), ylabel=r"$(R/L)_\mathrm{num}$")
        ax[1].set(xlabel=r"$[S_N\,\Lambda\,g(\tau)/4]^{1/3}$", ylabel="dev. (%)")
        ax[0].legend(loc="upper left")
        panel_label(ax[0], "(a)", x=-0.22); panel_label(ax[1], "(b)", x=-0.22)
        save(fig, "fig_collapse")

# ------------------------------------------------------------ Fig. invariant
if have("fig_invariant.csv"):
    d = pd.read_csv(f"{R}/fig_invariant.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(d.s_over_L, 100 * (d.invariant_norm - 1), "-", color=PALETTE["blue_main"])
    ax.axhline(0, color=PALETTE["red_strong"], ls="--", lw=1.8, label="exact (RFT)")
    ax.set(xlabel="$s/L$", ylabel=r"$T^\beta\sin\theta$, dev. from mean (%)", xlim=(0, 1))
    ax.legend(loc="upper left")
    save(fig, "fig_invariant")

# --------------------------------------------------------------- Fig. shapes
if have("fig_shapes.csv"):
    d = pd.read_csv(f"{R}/fig_shapes.csv")
    Ns = [2, 4, 8]
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.05]})
    lim = 1.1 * np.abs(d[["x", "y"]]).values.max()
    for i, N in enumerate(Ns):
        col, mk = N_STYLE[N]
        for j, x in d[d.N == N].groupby("thread"):
            ax[i].plot(x.x, x.z, "-", color=col, lw=1.8)
            tip = x.iloc[-1]
            ax[3].plot(tip.x, tip.y, mk, color=col, mfc="white", mew=1.8,
                       label=f"$N={N}$" if j == 0 else None)
        ax[i].set(xlim=(-lim, lim), xlabel="$x$ (m)", title=f"$N={N}$")
        ax[i].set_ylabel("$z$ (m)" if i == 0 else "")
        if i > 0:
            ax[i].tick_params(labelleft=False)
        panel_label(ax[i], "(" + "abc"[i] + ")", x=-0.12 if i else -0.3)
    ax[3].set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xlabel="$x$ (m)", ylabel="$y$ (m)")
    ax[3].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=11, handletextpad=0.2)
    panel_label(ax[3], "(d)", x=-0.3)
    save(fig, "fig_shapes")

# ----------------------------------------------------------- Fig. invariance
if have("fig_invariance.csv", "fig_invariance_t.csv"):
    d = pd.read_csv(f"{R}/fig_invariance.csv")
    t = pd.read_csv(f"{R}/fig_invariance_t.csv")
    relax = os.path.exists(f"{R}/relax.csv")
    n = 3 if relax else 2
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 3.8))
    for w, col, ls, lab in [(0.0, PALETTE["blue_main"], "-", "$w=0$"),
                            (0.5, PALETTE["red_strong"], "--", "$w=0.5$ m/s")]:
        for j, x in d[d.w == w].groupby("thread"):
            ax[0].plot(x.x_rel, x.z_rel, ls, color=col, lw=1.8, label=lab if j == 0 else None)
        y = t[t.w == w]
        ax[1].plot(y.t, y.z0, ls, color=col, label=lab)
    ax[0].set(xlabel=r"$x-x_0$ (m)", ylabel=r"$z-z_0$ (m)")
    ax[1].set(xlabel="$t$ (s)", ylabel="$z_0$ (m)")
    ax[0].legend(loc="lower right", fontsize=11); ax[1].legend(loc="upper left")
    panel_label(ax[0], "(a)"); panel_label(ax[1], "(b)")
    if relax:
        rr = pd.read_csv(f"{R}/relax.csv")
        ax[2].semilogy(rr.t, rr.dshape_over_L, "o-", color=PALETTE["blue_main"], mfc="white",
                       mew=1.8, label="shape")
        ax[2].semilogy(rr.t, rr.dR_over_L, "s--", color=PALETTE["teal"], mfc="white",
                       mew=1.8, label="$R$")
        ax[2].set(xlabel="$t$ (s)", ylabel="difference / $L$")
        ax[2].legend(loc="upper right")
        panel_label(ax[2], "(c)")
    save(fig, "fig_invariance")

# ------------------------------------------------------------- Fig. ksvalid
if have("fig_ksvalid.csv", "fig_ksvalid_pdf.csv"):
    s = pd.read_csv(f"{R}/fig_ksvalid.csv")
    p = pd.read_csv(f"{R}/fig_ksvalid_pdf.csv")
    Nk = s.N_k.max()
    s = s[(s.N_k == Nk) & (s.E_measured > 0)]
    kmax = 2 * np.pi / (5 * L / 100)          # resolution cut-off, N_t = 100
    s = s[s.k <= kmax]
    p = p[p.N_k == Nk]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].loglog(s.k, s.E_target, "-", color=PALETTE["red_strong"], label="von Kármán target")
    ax[0].loglog(s.k, s.E_measured, "o", color=PALETTE["blue_main"], mfc="white", mew=1.5,
                 ms=5, label=f"generated, $N_k={Nk}$")
    k0 = s.k.iloc[len(s) // 2]; e0 = s.E_target.iloc[len(s) // 2]
    kk = np.array([k0, 8 * k0])
    ax[0].loglog(kk, 2.5 * e0 * (kk / k0) ** (-5 / 3), "-", color=PALETTE["dark"], lw=1.5)
    ax[0].text(2.2 * k0, 2.5 * e0 * 2.2 ** (-5 / 3) * 1.4, "$-5/3$", fontsize=12)
    ax[0].set(xlabel="$k$ (1/m)", ylabel=r"$E_{11}(k)$ (m$^3$/s$^2$)")
    ax[0].legend(loc="lower left")
    ax[1].plot(p.w_bin_center, p.pdf, "o", color=PALETTE["blue_main"], mfc="white", mew=1.5, ms=5,
               label="generated")
    ax[1].plot(p.w_bin_center, p.gaussian, "-", color=PALETTE["red_strong"], label="Gaussian, $\\sigma_w$")
    ax[1].set(xlabel="$w$ (m/s)", ylabel="PDF")
    ax[1].legend(loc="upper left", bbox_to_anchor=(0.62, 1.0), fontsize=11)
    panel_label(ax[0], "(a)"); panel_label(ax[1], "(b)")
    save(fig, "fig_ksvalid")

# ------------------------------------------------------- Fig. phase (Stage B)
def P_dd(x, a=1.0, p0=0.25):
    """Eq. (eq:Pdd) in the scaled variable x = U0 h / K; a = K/K_eff, p0 = effective z0/h."""
    x = np.asarray(x, float) * a
    with np.errstate(all="ignore"):
        v = (1 - np.exp(-x * p0)) / (1 - np.exp(-x))
    return np.where(np.abs(x) < 1e-9, p0, v)


def fit_phase(d):
    """Maximum-likelihood fit of (a, p0) to all valid grid points (timeouts excluded)."""
    from scipy.optimize import minimize
    u, n, x = d.n_up.values, (d.n_up + d.n_down).values, d.x.values

    def nll(v):
        p = np.clip(P_dd(x, v[0], v[1]), 1e-12, 1 - 1e-12)
        return -(u * np.log(p) + (n - u) * np.log(1 - p)).sum()
    r = minimize(nll, [3.0, 0.3], bounds=[(0.5, 30), (0.05, 0.9)])
    p = P_dd(x, *r.x)
    chi2 = ((u - n * p) ** 2 / (n * p * (1 - p) + 1e-12)).sum()
    return r.x, chi2, len(d) - 2


if have("tab_phase.csv"):
    d = pd.read_csv(f"{R}/tab_phase.csv")
    (a_fit, p0_fit), chi2, dof = fit_phase(d)
    print(f"phase fit: a = {a_fit:.3f}, p0 = {p0_fit:.3f}, chi2 = {chi2:.1f} / {dof}")
    with open(os.path.join(O, "fig_phase_fit.txt"), "w") as f:
        f.write(f"a={a_fit:.4f}\np0={p0_fit:.4f}\nchi2={chi2:.2f}\ndof={dof}\n")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for N in sorted(d.N.unique()):
        col, mk = N_STYLE.get(N, (PALETTE["red_2"], "v"))
        for sw, fill in [(0.15, "white"), (0.22, PALETTE["neutral"]), (0.30, col)]:
            x = d[(d.N == N) & np.isclose(d.sigma_w, sw)]
            if x.empty:
                continue
            err = [np.maximum(0.0, x.P - x.P_lo), np.maximum(0.0, x.P_hi - x.P)]
            lab = f"$N={N}$" if sw == 0.30 or (N in (3, 6)) else None
            if sw == 0.30:
                ax[0].errorbar(x.Fbar_l, x.P, yerr=err, fmt=mk + "-", color=col, mfc=fill, mew=1.6,
                               ms=7, lw=1.4, capsize=2.5, label=lab)
            ax[1].errorbar(x.x + 0.06 * (np.log2(N) - 1.5), x.P, yerr=err, fmt=mk, color=col, mfc=fill,
                           mew=1.6, ms=7, capsize=2.5, label=lab if sw == 0.30 else None)
    ax[0].axvline(1, ls="--", color=PALETTE["dark"], lw=1.3)
    ax[0].set(xlabel=r"$\bar F_l$", ylabel="take-off probability $P$", ylim=(-0.03, 1.05),
              title=r"$\sigma_w=0.30$ m/s")
    ax[0].legend(loc="lower right", fontsize=11)
    xx = np.linspace(-4.3, 4.3, 400)
    ax[1].plot(xx, P_dd(xx), "--", color=PALETTE["red_strong"], lw=2, label=r"Eq. (Pdd), $K=\sigma_w\ell$")
    ax[1].plot(xx, P_dd(xx, a_fit, p0_fit), "-", color=PALETTE["dark"], lw=2,
               label=rf"fit: $K=\sigma_w\ell/{a_fit:.1f}$, $p_0={p0_fit:.2f}$")
    ax[1].set(xlabel=r"$x=w_s(\bar F_l-1)\,h/K$", ylim=(-0.03, 1.05))
    h, l = ax[1].get_legend_handles_labels()
    ax[1].legend(h[:2], l[:2], loc="upper left", fontsize=10.5)
    panel_label(ax[0], "(a)"); panel_label(ax[1], "(b)")
    save(fig, "fig_phase")

# ---------------------------------------------------- Fig. snapshot (Stage B)
import glob
snaps = sorted(glob.glob(os.path.join(R, "snapshots", "*.npz")))
if snaps and have("sweep.csv"):
    ke_ = ke
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    # (a) side view of one realization, three instants, nodes coloured by w
    z = None
    for f in snaps:
        zz = np.load(f, allow_pickle=True)
        if str(zz["outcome"]) == "rise":
            z = zz
            break
    if z is None:
        z = np.load(snaps[0], allow_pickle=True)
    t, X, U = z["t"], z["positions"], z["u"]
    idx = np.linspace(0, len(t) - 1, 4).round().astype(int)[1:]
    wmax = np.abs(U[idx, :, 2]).max()
    off = 0.0
    for k in idx:
        x0 = X[k, 0]
        sc = ax[0].scatter(X[k, :, 0] - x0[0] + off, X[k, :, 2], c=U[k, :, 2], cmap="RdBu_r",
                           vmin=-wmax, vmax=wmax, s=6, lw=0)
        ax[0].plot(off, x0[2], "o", color="black", ms=5)
        ax[0].text(off, X[k, :, 2].max() + 0.06, f"$t={t[k]:.1f}$ s", ha="center", fontsize=11)
        off += 0.45
    ztop = max(X[k, :, 2].max() for k in idx)
    ax[0].set_ylim(top=ztop + 0.2)
    ax[0].set(xlabel=r"$x-x_0$ (m), snapshots offset by 0.45 m", ylabel="$z$ (m)")
    cb = fig.colorbar(sc, ax=ax[0], fraction=0.05, pad=0.02)
    cb.set_label("$w$ at the nodes (m/s)")
    # (b) distribution of the run-averaged opening, N = 4, sigma = 0.30, x = 0
    sw = pd.read_csv(f"{R}/sweep.csv")
    sel = sw[(sw.N == 4) & np.isclose(sw.sigma_w, 0.30) & (sw.x == 0) & sw.valid]
    q = m * g / (4 * E)
    R_still = (S_N(4) * ke_ * q / (E * L**2) / 4) ** (1 / 3)   # Eq. (opening), tau = 1 -> g = 1
    ax[1].hist(sel.R_over_L_mean, bins=25, density=True, color=PALETTE["blue_secondary"], alpha=0.8,
               edgecolor="white", label=f"turbulence, $M={len(sel)}$")
    ax[1].axvline(R_still, color=PALETTE["red_strong"], ls="--", lw=2, label="still air, Eq. (opening)")
    ax[1].set(xlabel=r"run-averaged $R/L$", ylabel="PDF")
    ax[1].legend(loc="upper right", fontsize=10.5)
    panel_label(ax[0], "(a)", x=-0.12); panel_label(ax[1], "(b)")
    save(fig, "fig_snapshot")

print("done:", sorted(os.listdir(O)))
