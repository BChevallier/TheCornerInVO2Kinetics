"""Quality-control grids: every participant's fits drawn over their data.

For every data set (treadmill, bike tt1, bike tt2), one figure for the
cleaned (unfiltered) signal, one small panel per participant (in the order
of the prepared data file), showing:

- the 5-s VO2 samples as dots; phase I samples (5-15 s, excluded from the
  exponential fits) are hollow;
- the piecewise-linear fit (series 1), with a dotted line at its breakpoint;
- the phase II mono-exponential (series 2) from the iterative window: solid
  inside the window, dashed where extrapolated past `window_end` to 240 s,
  with a dotted line at `window_end`;
- the bi-exponential fit (series 3, thin), only where the slow component is
  supported for the cleaned signal (`exponential_model_selection`).

Panel titles give the participant id, breakpoint, phase II MRT and window
end. Panels whose phase II window stops at or before EARLY_WINDOW_S are
flagged ("!" prefix and a full outline): an early stop can mean a real slow
component or an artefact, so they are the ones to inspect.

Output: `analysis/figures/qc/<dataset>_fits_{light,dark}.{png,svg}`.

Run after both arms' steps 05-08: `python analysis/05_plot_participant_fits.py`.
"""

import math

import numpy as np
import pandas as pd

from analysis_common import DATASETS, FIGURES_DIR, REPO_ROOT, load_two_level, result_path
from biexponential_model import biexponential  # needs analysis_common imported first (sys.path)
from exponential_fitting import PHASE_I_CUTOFF_S
from monoexponential_model import monoexponential
from plot_style import THEMES, save_light_dark

import matplotlib.pyplot as plt  # noqa: E402  (after plot_style sets the Agg backend)
from matplotlib.lines import Line2D  # noqa: E402

METHOD = "cleaned"
EARLY_WINDOW_S = 70
N_COLUMNS = 7
PANEL_SIZE_IN = (2.0, 1.65)
T_END_S = 240

PREPARED_DATA = {
    "treadmill": REPO_ROOT / "treadmill" / "data" / "prepared" / "time_trial.csv",
    "bike_tt1": REPO_ROOT / "bike" / "data" / "prepared" / "tt1_time_trial.csv",
    "bike_tt2": REPO_ROOT / "bike" / "data" / "prepared" / "tt2_time_trial.csv",
}

T_DENSE = np.linspace(0, T_END_S, 4 * T_END_S + 1)


def load_dataset(dataset):
    """Data and every fit of the cleaned signal, one row per participant."""
    data = pd.read_csv(PREPARED_DATA[dataset])
    selection = pd.read_csv(result_path(dataset, "exponential_model_selection"))
    selection = selection[selection["method"] == METHOD].set_index("participant")
    return {
        "data": data,
        "participants": [col for col in data.columns if col != "time"],
        "piecewise": load_two_level(dataset, "piecewise_params")[METHOD],
        "phase2": load_two_level(dataset, "phase2_params")[METHOD],
        "biexp": load_two_level(dataset, "biexponential_params")[METHOD],
        "markers": load_two_level(dataset, "kinetics_markers")[METHOD],
        "slow_supported": selection["slow_component_supported"].astype(bool),
    }


def piecewise_curve(p):
    """The two segments as (t, VO2) arrays, each over its own interval."""
    bp = p["breakpoint"]
    t1 = np.array([0, bp])
    t2 = np.array([bp, T_END_S])
    return (np.concatenate([t1, t2[1:]]),
            np.concatenate([p["intercept1"] + p["slope1"] * t1,
                            p["intercept2"] + p["slope2"] * t2[1:]]))


def draw_panel(ax, t, y, participant, fits, colours):
    pw = fits["piecewise"].loc[participant]
    ph = fits["phase2"].loc[participant]
    bp, window_end = pw["breakpoint"], ph["window_end"]
    mrt = fits["markers"].loc[participant, "mrt"]
    early = window_end <= EARLY_WINDOW_S

    phase_i = (t > 0) & (t < PHASE_I_CUTOFF_S)
    ax.scatter(t[~phase_i], y[~phase_i], s=5, color=colours["muted"], linewidths=0, zorder=2)
    ax.scatter(t[phase_i], y[phase_i], s=6, facecolors="none", edgecolors=colours["muted"],
               linewidths=0.6, zorder=2)

    series = colours["series"]
    ax.plot(*piecewise_curve(pw), color=series[0], lw=1.3, zorder=4)
    ax.axvline(bp, color=series[0], lw=0.8, ls=":", zorder=3)

    mono = ph[["A0", "A1", "tau1", "TD1"]].to_numpy(dtype=float)
    inside = T_DENSE <= window_end
    ax.plot(T_DENSE[inside], monoexponential(T_DENSE[inside], *mono),
            color=series[1], lw=1.3, zorder=5)
    if window_end < T_END_S:
        beyond = T_DENSE >= window_end
        ax.plot(T_DENSE[beyond], monoexponential(T_DENSE[beyond], *mono),
                color=series[1], lw=1.1, ls=(0, (3, 2)), zorder=5)
        ax.axvline(window_end, color=series[1], lw=0.8, ls=":", zorder=3)

    if fits["slow_supported"].get(participant, False):
        bi = fits["biexp"].loc[participant].to_numpy(dtype=float)
        ax.plot(T_DENSE, biexponential(T_DENSE, *bi), color=series[2], lw=0.8, zorder=4.5)

    prefix = "! " if early else ""
    ax.set_title(f"{prefix}{participant}\nbp {bp:.0f} s · MRT {mrt:.0f} s · win {window_end:.0f} s",
                 fontsize=7, pad=3, color=colours["text"])
    ax.set_xlim(0, T_END_S)
    ax.set_xticks([0, 60, 120, 180, 240])
    ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    ax.grid(linewidth=0.4)
    if early:
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(colours["text_secondary"])
            ax.spines[side].set_linewidth(1.1)


def legend_handles(colours):
    series = colours["series"]
    return [
        Line2D([], [], ls="none", marker="o", ms=3, color=colours["muted"], mec="none",
               label="VO2 (5-s samples)"),
        Line2D([], [], ls="none", marker="o", ms=3.5, mfc="none", mec=colours["muted"], mew=0.7,
               label="phase I samples (not in exponential fits)"),
        Line2D([], [], color=series[0], lw=1.3, label="piecewise linear (dotted: breakpoint)"),
        Line2D([], [], color=series[1], lw=1.3, label="phase II mono-exp. in window"),
        Line2D([], [], color=series[1], lw=1.1, ls=(0, (3, 2)),
               label="phase II extrapolated (dotted: window end)"),
        Line2D([], [], color=series[2], lw=0.8, label="bi-exponential (slow comp. supported)"),
    ]


def figure_drawer(dataset, fits):
    participants = fits["participants"]
    t = fits["data"]["time"].to_numpy(dtype=float)
    n_rows = math.ceil(len(participants) / N_COLUMNS)

    def draw(theme):
        colours = THEMES[theme]
        fig, axes = plt.subplots(
            n_rows, N_COLUMNS, sharex=True, squeeze=False,
            figsize=(PANEL_SIZE_IN[0] * N_COLUMNS, PANEL_SIZE_IN[1] * n_rows + 0.8),
        )
        for i, ax in enumerate(axes.flat):
            if i >= len(participants):
                ax.set_visible(False)
                continue
            participant = participants[i]
            y = fits["data"][participant].to_numpy(dtype=float)
            draw_panel(ax, t, y, participant, fits, colours)
            row, col = divmod(i, N_COLUMNS)
            if col == 0:
                ax.set_ylabel("VO2 (L/min)", fontsize=7)
            if i + N_COLUMNS >= len(participants):
                ax.set_xlabel("time (s)", fontsize=7)
                ax.tick_params(labelbottom=True)
        n_flagged = int((fits["phase2"].loc[participants, "window_end"] <= EARLY_WINDOW_S).sum())
        fig.suptitle(
            f"{dataset}: fits to the unfiltered signal, n = {len(participants)} "
            f"(! = phase II window ends at or before {EARLY_WINDOW_S} s: {n_flagged})",
            fontsize=10, color=colours["text"], y=1.0,
        )
        fig.legend(handles=legend_handles(colours), loc="lower center", ncol=3, fontsize=7.5,
                   bbox_to_anchor=(0.5, -0.005))
        fig.tight_layout(rect=(0, 0.03, 1, 0.985), h_pad=0.6, w_pad=0.4)
        return fig

    return draw


def main():
    out_dir = FIGURES_DIR / "qc"
    for dataset in DATASETS:
        fits = load_dataset(dataset)
        written = save_light_dark(figure_drawer(dataset, fits), out_dir / f"{dataset}_fits")
        print(f"{dataset}: {len(written)} files in {out_dir}")


if __name__ == "__main__":
    main()
