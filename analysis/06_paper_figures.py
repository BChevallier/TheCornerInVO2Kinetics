"""Paper (and slide) figures, each in a light and a dark version.

Everything is drawn from the arms' result files and the tables in
`analysis/results/`; nothing is refitted except evaluating the stored
piecewise and phase II parameters. Unfiltered (`cleaned`) signal unless
stated. Series colours: 1 = treadmill, 2 = bike tt1, 3 = bike tt2; in
method comparisons 1 = piecewise, 2 = phase II mono-exponential.

- fig1_example: one treadmill participant, LB0303 — the data, the piecewise
  fit with its breakpoint, the phase II exponential (solid inside its
  fitting window, dashed beyond) and TD, MRT and breakpoint on the time
  axis. LB0303 was chosen because its breakpoint / MRT (1.585) is the
  treadmill median (1.584) and its phase II window is long (to 180 s, the
  treadmill median is 115 s), so the exponential is not a short early fit;
  its tau (21.6 s) and TD (11.8 s) are typical of the treadmill group.
- fig2_breakpoint_vs_mrt: breakpoint vs phase II MRT, one panel per data
  set, least-squares line, r [95% CI] and n from `validity.csv`, and
  breakpoint = 1.6 x MRT for reference.
- fig3_simulation: noise-free simulated breakpoint / MRT vs tau, one line
  per TD (`simulation_grid.csv`, reference amplitudes), with the observed
  participants (breakpoint / MRT vs fitted tau1) and the pooled observed
  median and interquartile range.
- fig4_bland_altman: bike tt2 - tt1 vs mean for the breakpoint and phase II
  MRT, bias and 95% limits of agreement from `reliability.csv`.
- fig5_reliability_icc: ICC(3,1) with 95% CI for every quantity in
  `reliability.csv`, sorted.
- fig6_smoothing: within-participant differences smoothed - unfiltered
  (Savitzky-Golay, Butterworth) for the breakpoint and phase II tau.

Outputs: `analysis/figures/<stem>_{light,dark}.{png,svg}`.

Run after 01-04: `python analysis/06_paper_figures.py`.
"""

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from analysis_common import (
    DATASETS, FIGURES_DIR, REPO_ROOT, RESULTS_DIR, load_breakpoints, load_two_level, result_path,
)
from monoexponential_model import monoexponential  # needs analysis_common imported first (sys.path)
from plot_style import THEMES, save_light_dark  # needs analysis_common imported first (sys.path)

LABELS = {"treadmill": "Treadmill", "bike_tt1": "Bike TT1", "bike_tt2": "Bike TT2"}
SERIES = {"treadmill": 0, "bike_tt1": 1, "bike_tt2": 2}
PREPARED = {
    "treadmill": REPO_ROOT / "treadmill" / "data" / "prepared" / "time_trial.csv",
    "bike_tt1": REPO_ROOT / "bike" / "data" / "prepared" / "tt1_time_trial.csv",
    "bike_tt2": REPO_ROOT / "bike" / "data" / "prepared" / "tt2_time_trial.csv",
}
EXAMPLE = ("treadmill", "LB0303")
REFERENCE_RATIO = 1.6
METHOD = "cleaned"


def font_sizes(base):
    """Larger fonts than the rc default; set inside `draw`, so the outer
    theme rc_context restores them afterwards."""
    plt.rcParams.update({
        "font.size": base, "axes.titlesize": base + 1, "axes.labelsize": base,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
    })


def blend(color, surface, amount):
    """`color` moved `amount` of the way towards `surface` (0 = unchanged)."""
    c, s = np.array(mcolors.to_rgb(color)), np.array(mcolors.to_rgb(surface))
    return tuple(c + amount * (s - c))


def per_participant(dataset):
    """breakpoint, MRT, tau1, TD1 and phase II window end per participant."""
    phase2 = load_two_level(dataset, "phase2_params")[METHOD]
    return pd.DataFrame({
        "bp": load_breakpoints(dataset)[METHOD],
        "mrt": load_two_level(dataset, "kinetics_markers")[METHOD]["mrt"],
        "tau": phase2["tau1"], "td": phase2["TD1"], "window_end": phase2["window_end"],
    }).astype(float)


# ---------------------------------------------------------------- figure 1

def draw_example(theme):
    t = THEMES[theme]
    font_sizes(15)
    dataset, pid = EXAMPLE
    data = pd.read_csv(PREPARED[dataset])
    x, y = data["time"].values.astype(float), data[pid].values.astype(float)
    pw = load_two_level(dataset, "piecewise_params")[METHOD].loc[pid].astype(float)
    p2 = load_two_level(dataset, "phase2_params")[METHOD].loc[pid].astype(float)
    bp = pw["breakpoint"]
    td, tau, A0, A1, end = p2["TD1"], p2["tau1"], p2["A0"], p2["A1"], p2["window_end"]
    mrt = td + tau
    c_pw, c_exp = t["series"][0], t["series"][1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, "o", ms=5, color=t["muted"], alpha=0.8, mec="none", label="VO$_2$ (5-s data)",
            zorder=2)
    xs = np.array([x[0], bp, x[-1]])
    ys = np.where(xs <= bp, pw["intercept1"] + pw["slope1"] * xs, pw["intercept2"] + pw["slope2"] * xs)
    ax.plot(xs, ys, color=c_pw, lw=2.4, label="Piecewise linear fit", zorder=4)
    inside = np.linspace(0, end, 400)
    beyond = np.linspace(end, x[-1], 200)
    ax.plot(inside, monoexponential(inside, A0, A1, tau, td), color=c_exp, lw=2.4,
            label="Phase II exponential, fitting window\n(MRT = TD + τ)", zorder=3)
    ax.plot(beyond, monoexponential(beyond, A0, A1, tau, td), color=c_exp, lw=2.0, ls=(0, (4, 3)),
            label="Phase II extrapolated", zorder=3)

    ymin = 0.0
    ax.set_ylim(ymin, max(y.max(), ys.max()) * 1.08)
    ax.set_xlim(0, x[-1] + 2)
    bp_y = pw["intercept1"] + pw["slope1"] * bp
    mrt_y = monoexponential(mrt, A0, A1, tau, td)
    marks = [
        (td, A0, f"TD\n{td:.0f} s", c_exp),
        (mrt, mrt_y, f"MRT\n{mrt:.0f} s", c_exp),
        (bp, bp_y, f"Breakpoint\n{bp:.0f} s", c_pw),
    ]
    for xm, ym, text, color in marks:
        ax.plot([xm, xm], [ymin, ym], color=t["text_secondary"], lw=1, ls=":", zorder=1)
        ax.plot(xm, ym, "o", ms=9, mfc=t["surface"], mec=color, mew=2.2, zorder=6)
        ax.text(xm + 1.2, 0.62, text, color=t["text"], fontsize=13, va="center", ha="left",
                linespacing=1.15)
    ax.annotate("", xy=(bp, 0.16), xytext=(mrt, 0.16),
                arrowprops=dict(arrowstyle="<->", color=t["text_secondary"], lw=1.2,
                                shrinkA=0, shrinkB=0))
    ax.text(bp + 1.2, 0.16, f"breakpoint = {bp / mrt:.2f} × MRT", ha="left", va="center",
            color=t["text_secondary"], fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("VO$_2$ (L/min)")
    ax.legend(loc="lower right", handlelength=2.6)
    ax.set_title(f"4-min maximal bout, treadmill participant {pid}", loc="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- figure 2

def draw_bp_vs_mrt(theme):
    t = THEMES[theme]
    font_sizes(13)
    validity = pd.read_csv(RESULTS_DIR / "validity.csv")
    validity = validity[(validity["method"] == METHOD) & (validity["marker"] == "mrt")]
    tables = {d: per_participant(d).dropna(subset=["bp", "mrt"]) for d in DATASETS}
    all_mrt = pd.concat([tb["mrt"] for tb in tables.values()])
    all_bp = pd.concat([tb["bp"] for tb in tables.values()])
    xlim = (np.floor(all_mrt.min()) - 2, np.ceil(all_mrt.max()) + 2)
    ylim = (np.floor(all_bp.min() / 10) * 10 - 2, np.ceil(all_bp.max() / 10) * 10 + 2)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True, sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        tb = tables[dataset]
        row = validity[validity["dataset"] == dataset].iloc[0]
        color = t["series"][SERIES[dataset]]
        line_x = np.array(xlim)
        ax.plot(line_x, REFERENCE_RATIO * line_x, color=t["muted"], lw=1.2, ls=(0, (5, 4)),
                label=f"Breakpoint = {REFERENCE_RATIO} × MRT", zorder=1)
        fit_x = np.array([tb["mrt"].min(), tb["mrt"].max()])
        ax.plot(fit_x, row["intercept"] + row["slope"] * fit_x, color=color, lw=2,
                label="Least-squares line", zorder=3)
        ax.plot(tb["mrt"], tb["bp"], "o", ms=6, color=color, alpha=0.75, mec="none", zorder=2)
        ax.text(0.04, 0.96,
                f"r = {row['r']:.2f} [{row['r_ci_low']:.2f}, {row['r_ci_high']:.2f}]\n"
                f"n = {int(row['n'])}",
                transform=ax.transAxes, va="top", ha="left", color=t["text"], fontsize=12)
        ax.set_title(LABELS[dataset], loc="left")
        ax.set_xlabel("Phase II MRT (s)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    axes[0].set_ylabel("Piecewise breakpoint (s)")
    handles = [
        Line2D([], [], color=t["text_secondary"], lw=2, label="Least-squares line"),
        Line2D([], [], color=t["muted"], lw=1.2, ls=(0, (5, 4)),
               label=f"Breakpoint = {REFERENCE_RATIO} × MRT"),
    ]
    axes[2].legend(handles=handles, loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- figure 3

def draw_simulation(theme):
    t = THEMES[theme]
    font_sizes(13)
    grid = pd.read_csv(RESULTS_DIR / "simulation_grid.csv")
    grid = grid[(grid["A0"] == 1.0) & (grid["A1"] == 3.0)]
    observed = {d: per_participant(d).dropna(subset=["bp", "mrt", "tau"]) for d in DATASETS}
    ratios = pd.concat([tb["bp"] / tb["mrt"] for tb in observed.values()])
    q1, med, q3 = np.percentile(ratios, [25, 50, 75])

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.axhspan(q1, q3, color=t["muted"], alpha=0.18, lw=0, zorder=0)
    ax.axhline(med, color=t["muted"], lw=1, ls=(0, (5, 4)), zorder=1)
    band = (
        mpatches.Patch(color=t["muted"], alpha=0.25, lw=0),
        Line2D([], [], color=t["muted"], lw=1, ls=(0, (5, 4))),
    )

    tds = sorted(grid["TD"].unique())
    # neutral ramp (simulation) so the data-set colours stay for the observed points;
    # sawtooth = the breakpoint jumping between 5-s samples
    sim_handles = []
    for i, td in enumerate(tds):
        g = grid[grid["TD"] == td].sort_values("tau")
        shade = blend(t["text"], t["surface"], 0.15 + 0.6 * (1 - i / (len(tds) - 1)))
        (line,) = ax.plot(g["tau"], g["bp_over_mrt"], color=shade, lw=1.6, zorder=2,
                          label=f"TD {td:.0f} s")
        sim_handles.append(line)

    obs_handles = []
    for dataset, tb in observed.items():
        (pts,) = ax.plot(tb["tau"], tb["bp"] / tb["mrt"], "o", ms=5.5, alpha=0.8, mec="none",
                         color=t["series"][SERIES[dataset]], label=LABELS[dataset], zorder=3)
        obs_handles.append(pts)
    ax.set_xlim(0, 52)
    ax.set_xlabel("Phase II τ (s)")
    ax.set_ylabel("Breakpoint / MRT")
    blank = Line2D([], [], lw=0)
    handles = ([blank] + sim_handles + [blank, blank] + obs_handles
               + [band])
    labels = (["Simulated, noise-free"] + [h.get_label() for h in sim_handles]
              + ["", "Observed (unfiltered)"] + [h.get_label() for h in obs_handles]
              + [f"Observed median {med:.2f},\nIQR {q1:.2f}–{q3:.2f}"])
    legend = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                       handlelength=2.2, borderaxespad=0)
    for text, label in zip(legend.get_texts(), labels):
        if label in ("Simulated, noise-free", "Observed (unfiltered)"):
            text.set_color(t["text"])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- figure 4

def draw_bland_altman(theme):
    t = THEMES[theme]
    font_sizes(13)
    reliability = pd.read_csv(RESULTS_DIR / "reliability.csv")
    reliability = reliability[reliability["method"] == METHOD].set_index("quantity")
    tt1, tt2 = per_participant("bike_tt1"), per_participant("bike_tt2")
    panels = [("bp", "breakpoint", "Piecewise breakpoint", 0),
              ("mrt", "mrt", "Phase II MRT", 1)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (col, quantity, title, series) in zip(axes, panels):
        pair = pd.DataFrame({"a": tt1[col], "b": tt2[col]}).dropna()
        mean, diff = (pair["a"] + pair["b"]) / 2, pair["b"] - pair["a"]
        row = reliability.loc[quantity]
        ax.plot(mean, diff, "o", ms=6.5, alpha=0.8, mec="none", color=t["series"][series], zorder=3)
        ax.axhline(0, color=t["axis"], lw=1, zorder=1)
        ax.axhline(row["bias"], color=t["text_secondary"], lw=1.4, zorder=2)
        for level in (row["loa_low"], row["loa_high"]):
            ax.axhline(level, color=t["muted"], lw=1.2, ls=(0, (5, 4)), zorder=2)
        lo, hi = mean.min(), mean.max()
        pad = 0.08 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad * 6)
        right = hi + pad * 5.8
        for level, label in [(row["loa_high"], f"+1.96 SD\n{row['loa_high']:+.1f} s"),
                             (row["bias"], f"Bias\n{row['bias']:+.1f} s"),
                             (row["loa_low"], f"−1.96 SD\n{row['loa_low']:+.1f} s")]:
            ax.text(right, level, label.replace("-", "−"), ha="right", va="center",
                    color=t["text_secondary"], fontsize=11, linespacing=1.1,
                    bbox=dict(boxstyle="square,pad=0.15", fc=t["surface"], ec="none"))
        ax.set_title(f"{title}  (ICC {row['icc_3_1']:.2f}, TE {row['typical_error']:.1f} s)",
                     loc="left")
        ax.set_xlabel("Mean of TT1 and TT2 (s)")
    axes[0].set_ylabel("TT2 − TT1 (s)")
    lim = max(abs(reliability.loc[["breakpoint", "mrt"], ["loa_low", "loa_high"]]).max().max(),
              max((tt2[c] - tt1[c]).abs().max() for c in ("bp", "mrt"))) * 1.12
    axes[0].set_ylim(-lim, lim)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- figure 5

ICC_LABELS = {
    "breakpoint": "Piecewise breakpoint", "mrt": "Phase II MRT",
    "mrt_full_window": "MRT, whole-trial fit", "tau": "Phase II τ", "td": "Phase II TD",
    "fast_end_95": "Phase II 95% time", "slow_component": "Slow component amplitude",
    "t50": "t50 (model-free)", "t90": "t90 (model-free)", "td2_supported": "TD2, slow-component onset",
}


def draw_icc(theme):
    t = THEMES[theme]
    font_sizes(13)
    rel = pd.read_csv(RESULTS_DIR / "reliability.csv")
    rel = rel[rel["method"] == METHOD].sort_values("icc_3_1").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for i, row in rel.iterrows():
        highlight = row["quantity"] == "breakpoint"
        color = t["series"][0] if highlight else t["muted"]
        ax.plot([row["icc_ci_low"], row["icc_ci_high"]], [i, i], color=color,
                lw=2.4 if highlight else 1.6, solid_capstyle="butt")
        ax.plot(row["icc_3_1"], i, "o", ms=9 if highlight else 7, color=color, zorder=3)
        ax.text(1.02, i, f"{row['icc_3_1']:.2f}", va="center", ha="left", fontsize=12,
                color=t["text"] if highlight else t["text_secondary"],
                fontweight="bold" if highlight else "normal")
    labels = [ICC_LABELS[q] + (f" (n = {n})" if n != rel["n"].max() else "")
              for q, n in zip(rel["quantity"], rel["n"])]
    ax.set_yticks(range(len(rel)), labels)
    for tick, q in zip(ax.get_yticklabels(), rel["quantity"]):
        if q == "breakpoint":
            tick.set_color(t["text"])
            tick.set_fontweight("bold")
    ax.axvline(0, color=t["axis"], lw=1, zorder=0)
    ax.set_xlim(-0.55, 1.0)
    ax.set_ylim(-0.6, len(rel) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"ICC(3,1), bike TT1 vs TT2, 95% CI (n = {rel['n'].max()})")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- figure 6

def smoothing_differences():
    """Long table: dataset, quantity, variant, difference (variant - cleaned)."""
    rows = []
    for dataset in DATASETS:
        tables = {
            "breakpoint": load_breakpoints(dataset),
            "tau": pd.DataFrame({v: load_two_level(dataset, "phase2_params")[v]["tau1"]
                                 for v in ("cleaned", "sg", "bw")}),
        }
        for quantity, tb in tables.items():
            tb = tb[["cleaned", "sg", "bw"]].astype(float).dropna()
            for variant in ("sg", "bw"):
                for value in tb[variant] - tb["cleaned"]:
                    rows.append((dataset, quantity, variant, value))
    return pd.DataFrame(rows, columns=["dataset", "quantity", "variant", "diff"])


def draw_smoothing(theme):
    t = THEMES[theme]
    font_sizes(13)
    diffs = smoothing_differences()
    rng = np.random.default_rng(6)
    jitter = {key: rng.uniform(-0.09, 0.09, len(g))
              for key, g in diffs.groupby(["dataset", "quantity", "variant"], sort=True)}
    quantities = [("breakpoint", "Piecewise breakpoint"), ("tau", "Phase II τ")]
    variants = [("sg", "Savitzky-Golay − unfiltered"), ("bw", "Butterworth − unfiltered")]
    offsets = {"treadmill": -0.27, "bike_tt1": 0.0, "bike_tt2": 0.27}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (variant, title) in zip(axes, variants):
        ax.axhline(0, color=t["text_secondary"], lw=1, zorder=1)
        for qi, (quantity, _) in enumerate(quantities):
            for dataset in DATASETS:
                values = diffs[(diffs["dataset"] == dataset) & (diffs["quantity"] == quantity)
                               & (diffs["variant"] == variant)]["diff"].values
                pos = qi + offsets[dataset]
                color = t["series"][SERIES[dataset]]
                ax.boxplot(values, positions=[pos], widths=0.2, showfliers=False,
                           medianprops=dict(color=t["text"], lw=1.6),
                           boxprops=dict(color=t["text_secondary"], lw=1),
                           whiskerprops=dict(color=t["text_secondary"], lw=1),
                           capprops=dict(color=t["text_secondary"], lw=1), zorder=2)
                ax.plot(pos + jitter[(dataset, quantity, variant)], values, "o", ms=4,
                        alpha=0.7, mec="none", color=color, zorder=3,
                        label=LABELS[dataset] if qi == 0 else None)
        ax.set_xticks(range(len(quantities)), [label for _, label in quantities])
        ax.set_xlim(-0.55, len(quantities) - 0.45)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Within-participant difference (s)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    return fig


FIGURES = {
    "fig1_example": draw_example,
    "fig2_breakpoint_vs_mrt": draw_bp_vs_mrt,
    "fig3_simulation": draw_simulation,
    "fig4_bland_altman": draw_bland_altman,
    "fig5_reliability_icc": draw_icc,
    "fig6_smoothing": draw_smoothing,
}


def main():
    for stem, draw in FIGURES.items():
        for path in save_light_dark(draw, FIGURES_DIR / stem):
            print(f"saved {path}")


if __name__ == "__main__":
    main()
