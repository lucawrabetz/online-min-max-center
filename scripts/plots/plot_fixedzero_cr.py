"""unitsquarefixedzero full grid: rows = arrival orders, cols = (objective, facilities,
competitive ratio vs OMIP). Plus a CCT cross-order CR panel at the foot of the CR column.

Run from repo root: ./okc/bin/python scripts/plots/plot_fixedzero_cr.py
Writes out/baselines_fixedzero_cr.pdf / .png

NOTE on the shaded band: it is the 95% CI of the MEAN (t*std/sqrt(30)), NOT per-instance
spread. Bands look invisible on CCT none/farthest because CI is ~1-2% of the value there,
not because variance is zero (per-instance obj std ~0.7-1.0). To show per-instance spread
instead, swap the CI half-width for +/-1 std or a 10-90 percentile envelope in stats()/cr_stats().
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

C_CCTA, C_BEA, C_NKCA = "#2a78d6", "#eb6834", "#1baf7a"
C_OMIP, C_SOMIP = "#4a3aa7", "#5c5b55"
ONLINE = {"CCTA": C_CCTA, "BEA": C_BEA, "NKCA": C_NKCA}
# arrival-order difficulty ramp (light->dark = easy->hard), distinct from algo colors
ORDER_C = {"nearest": "#c2a5cf", "none": "#9970ab", "farthest": "#762a83"}
TEXT, MUTED, GRID = "#1a1a19", "#5c5b55", "#dededa"
TCRIT = 2.045  # t_0.975, dof=29

df = pd.read_csv("out/final.csv")
d = df[df.set_name == "unitsquarefixedzero"]


def stats(sub, col):
    g = sub.groupby("Gamma_run")[col]
    m, sd, n = g.mean(), g.std(ddof=1), g.count()
    return m, TCRIT * sd / np.sqrt(n.where(n > 1))


def cr_stats(dp, algo):
    """Per-instance competitive ratio obj_algo/obj_OMIP, then mean +/- CI over instances."""
    key = ["Gamma_run", "id"]
    a = dp[dp.solver == algo][key + ["objective"]].rename(columns={"objective": "a"})
    o = dp[dp.solver == "OMIP"][key + ["objective"]].rename(columns={"objective": "o"})
    m = a.merge(o, on=key)
    m = m[m.o > 0]  # drop Gamma=0 (0/0)
    m["cr"] = m["a"] / m["o"]
    g = m.groupby("Gamma_run")["cr"]
    mean, sd, n = g.mean(), g.std(ddof=1), g.count()
    return mean, TCRIT * sd / np.sqrt(n.where(n > 1))


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
    ax.set_xlim(0, 1.4)


def band(ax, gam, m, e, color, **kw):
    ax.fill_between(
        gam,
        (m - e).reindex(gam),
        (m + e).reindex(gam),
        color=color,
        alpha=0.16,
        lw=0,
        zorder=2,
    )
    ax.plot(gam, m.reindex(gam), color=color, zorder=3, **kw)


PERMS = ["none", "nearest", "farthest"]
PERM_TITLE = {
    "none": "random order (none)",
    "nearest": "nearest arrival (easy)",
    "farthest": "farthest arrival (hard)",
}

fig = plt.figure(figsize=(16.5, 15.2))
gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 1], hspace=0.42, wspace=0.24)

for r, perm in enumerate(PERMS):
    dp = d[d.perm == perm]
    axo = fig.add_subplot(gs[r, 0])
    axf = fig.add_subplot(gs[r, 1])
    axc = fig.add_subplot(gs[r, 2])
    for ax in (axo, axf, axc):
        style(ax)
        ax.set_xlabel(r"fixed cost $\Gamma$", color=TEXT, fontsize=9.5)

    # col 0: objective + OMIP + SOMIP
    for name, c in ONLINE.items():
        sub = dp[dp.solver == name]
        gam = sorted(sub.Gamma_run.unique())
        band(axo, gam, *stats(sub, "objective"), color=c, lw=1.8)
    for name, c, ls, mk in [
        ("OMIP", C_OMIP, (0, (5, 2)), None),
        ("SOMIP", C_SOMIP, (0, (1, 1.6)), "D"),
    ]:
        sub = dp[dp.solver == name]
        gam = sorted(sub.Gamma_run.unique())
        m, e = stats(sub, "objective")
        axo.fill_between(
            gam,
            (m - e).reindex(gam),
            (m + e).reindex(gam),
            color=c,
            alpha=0.12,
            lw=0,
            zorder=1,
        )
        axo.plot(
            gam,
            m.reindex(gam),
            color=c,
            lw=1.8,
            ls=ls,
            zorder=4,
            marker=mk,
            ms=3,
            mec="none",
        )

    # col 1: facilities (online only)
    for name, c in ONLINE.items():
        sub = dp[dp.solver == name]
        gam = sorted(sub.Gamma_run.unique())
        band(axf, gam, *stats(sub, "num_facilities"), color=c, lw=1.8)

    # col 2: competitive ratio vs OMIP (3 online algos)
    axc.axhline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    for name, c in ONLINE.items():
        m, e = cr_stats(dp, name)
        gam = sorted(m.index)
        band(axc, gam, m, e, color=c, lw=1.8)

    axo.set_ylabel("objective", color=TEXT, fontsize=10)
    axf.set_ylabel(r"$|F_T|$ (incl. $x_0$)", color=TEXT, fontsize=10)
    axc.set_ylabel("competitive ratio (obj / OMIP)", color=TEXT, fontsize=10)
    axo.set_title(
        f"objective — {PERM_TITLE[perm]}", color=TEXT, fontsize=10.5, loc="left", pad=8
    )
    axf.set_title(
        f"facilities — {PERM_TITLE[perm]}", color=TEXT, fontsize=10.5, loc="left", pad=8
    )
    axc.set_title(
        f"competitive ratio — {PERM_TITLE[perm]}",
        color=TEXT,
        fontsize=10.5,
        loc="left",
        pad=8,
    )

# 4th CR plot: CCT competitive ratio across the three arrival orders
axcc = fig.add_subplot(gs[3, 2])
style(axcc)
axcc.set_xlabel(r"fixed cost $\Gamma$", color=TEXT, fontsize=9.5)
axcc.axhline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
for perm in PERMS:
    m, e = cr_stats(d[d.perm == perm], "CCTA")
    gam = sorted(m.index)
    band(axcc, gam, m, e, color=ORDER_C[perm], lw=2)
axcc.set_ylabel("CCT competitive ratio", color=TEXT, fontsize=10)
axcc.set_title(
    "CCT competitive ratio across arrival orders",
    color=TEXT,
    fontsize=10.5,
    loc="left",
    pad=8,
)
oleg = axcc.legend(
    handles=[
        Line2D(
            [0],
            [0],
            color=ORDER_C[p],
            lw=2,
            label={
                "none": "random (none)",
                "nearest": "nearest (easy)",
                "farthest": "farthest (hard)",
            }[p],
        )
        for p in PERMS
    ],
    frameon=False,
    fontsize=8.5,
    loc="upper left",
)
for t in oleg.get_texts():
    t.set_color(TEXT)

# shared top legend (algos + offline references)
handles = [
    Line2D([0], [0], color=C_CCTA, lw=2, label="CCT"),
    Line2D([0], [0], color=C_BEA, lw=2, label="Break-even"),
    Line2D([0], [0], color=C_NKCA, lw=2, label="Naive $k$-center"),
    Line2D(
        [0], [0], color=C_OMIP, lw=2, ls=(0, (5, 2)), label="OMIP (fully-offline OPT)"
    ),
    Line2D(
        [0],
        [0],
        color=C_SOMIP,
        lw=2,
        ls=(0, (1, 1.6)),
        marker="D",
        ms=4,
        mec="none",
        label="SOMIP (semi-offline)",
    ),
]
leg = fig.legend(
    handles=handles,
    ncol=5,
    frameon=False,
    fontsize=10,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.988),
)
for t in leg.get_texts():
    t.set_color(TEXT)

fig.suptitle(
    "unitsquarefixedzero ($n=2$, $T=50$, 30 instances) — fine $\\Gamma$ sweep, 95% CI",
    color=TEXT,
    fontsize=13,
    x=0.02,
    ha="left",
    y=0.998,
)
fig.subplots_adjust(top=0.94, left=0.05, right=0.985, bottom=0.045)
fig.savefig("out/baselines_fixedzero_cr.pdf", facecolor="white")
fig.savefig("out/baselines_fixedzero_cr.png", dpi=140, facecolor="white")
print("wrote out/baselines_fixedzero_cr.pdf / .png")
