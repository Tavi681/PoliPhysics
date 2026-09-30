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
           4: (PALETTE["teal"], "s"), 8: (PALETTE["violet"], "^")}

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

print("done:", sorted(os.listdir(O)))
