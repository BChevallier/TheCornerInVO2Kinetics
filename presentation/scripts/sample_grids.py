"""Slide figures: every participant's measured VO2, one small panel each.

Slide version of the QC grids (analysis/05_plot_participant_fits.py) with
the data only: no fits, no phase labels, no participant IDs (not introduced
yet in the talk). The two example participants (vo2_examples.py) are
outlined in orange. Each panel uses the frame of the "actually" examples:
time 0-4 min, unfiltered 5-s values scaled to the person's own maximum, so
panels compare shapes, not absolute VO2.

Writes presentation/figures/grid_{treadmill,bike_tt1,bike_tt2}_{light,dark}.{png,svg}.
Run from anywhere with the project venv.
"""

import math

import pandas as pd

from vo2_domains import OUT_DIR, REPO_ROOT, THEMES, plt, save_light_dark
from vo2_examples import EXAMPLES

DATASETS = {
    "treadmill": (REPO_ROOT / "treadmill/data/prepared/time_trial.csv", 8),
    "bike_tt1": (REPO_ROOT / "bike/data/prepared/tt1_time_trial.csv", 7),
    "bike_tt2": (REPO_ROOT / "bike/data/prepared/tt2_time_trial.csv", 7),
}


def draw(theme, name):
    t = THEMES[theme]
    path, ncols = DATASETS[name]
    data = pd.read_csv(path)
    pids = [c for c in data.columns if c != "time"]
    nrows = math.ceil(len(pids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 6.6), sharex=True, sharey=True)
    for ax in axes.flat[len(pids):]:
        ax.set_visible(False)
    for ax, pid in zip(axes.flat, pids):
        ax.plot(data["time"] / 60, data[pid] / data[pid].max(), "o", ms=2.6,
                color=t["series"][0], markeredgewidth=0)
        ax.grid(False)
        if (path, pid) in EXAMPLES.values():
            # the runner / cyclist shown on the slide before: outlined in orange
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(t["series"][1])
                spine.set_linewidth(2.5)
    ax0 = axes.flat[0]
    ax0.set_xlim(-0.2, 4.2)
    ax0.set_ylim(0, 1.15)
    ax0.set_xticks(range(0, 5))
    ax0.set_yticks([])
    for i, ax in enumerate(axes.flat):
        # time labels on the lowest *visible* panel of each column
        lowest = i + ncols >= len(pids)
        ax.tick_params(labelsize=10, labelbottom=lowest)
    fig.supxlabel("Time (min)", fontsize=14, color=t["text_secondary"])
    fig.supylabel("Oxygen uptake", fontsize=14, color=t["text_secondary"])
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    return fig


if __name__ == "__main__":
    for name in DATASETS:
        for path in save_light_dark(lambda theme: draw(theme, name), OUT_DIR / f"grid_{name}"):
            print(path.relative_to(REPO_ROOT))
