"""Slide figure: age distribution of the analysed sample, women vs men.

One dot per participant, stacked in 2-year age bins (a dot histogram reads
as "people" for a lay audience), runners and cyclists side by side on the
same age axis. Only participants in the analysis are shown: the treadmill
runners with a piecewise fit, the 27 cyclists left after exclusions.

Writes presentation/figures/sample_ages_{light,dark}.{png,svg}.
Run from anywhere: `python presentation/scripts/sample_ages.py`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "shared"))
import matplotlib.pyplot as plt  # noqa: E402
from slide_style import THEMES, save_light_dark  # noqa: E402  (black dark theme)

OUT_STEM = REPO_ROOT / "presentation" / "figures" / "sample_ages"
BIN_WIDTH = 2  # years
AGE_RANGE = (16, 48)
X_LIMITS = (15, 49)  # half a bin of margin so edge dots are not clipped

# The treadmill time-trial file calls one runner ZM311; general_data.csv has
# ZM3103 and no ZM311 (every other ID is 6 characters) -> same person, typo.
TREADMILL_ID_FIXES = {"ZM311": "ZM3103"}


def load_samples():
    general = pd.read_csv(REPO_ROOT / "treadmill/data/raw/general_data.csv")
    runners = pd.read_csv(REPO_ROOT / "treadmill/results/piecewise_breakpoints.csv",
                          index_col=0).index
    runners = runners.map(lambda pid: TREADMILL_ID_FIXES.get(pid, pid))
    treadmill = general[general["ID"].isin(runners)]
    assert len(treadmill) == len(runners), "unmatched runner IDs"

    master = pd.read_csv(REPO_ROOT / "bike/data/master_table.csv")
    riders = pd.read_csv(REPO_ROOT / "bike/results/tt1_piecewise_breakpoints.csv",
                         index_col=0).index
    bike = master[master["pid"].str.lower().isin(riders)]
    assert len(bike) == len(riders), "unmatched cyclist IDs"
    return {"Runners": treadmill[["sex", "age"]], "Cyclists": bike[["sex", "age"]]}


def draw(theme, samples):
    t = THEMES[theme]
    colours = {"female": t["series"][0], "male": t["series"][1]}
    edges = np.arange(AGE_RANGE[0], AGE_RANGE[1] + BIN_WIDTH, BIN_WIDTH)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    max_stack = 0
    for ax, (name, df) in zip(axes, samples.items()):
        bins = np.digitize(df["age"], edges) - 1
        for b in np.unique(bins):
            in_bin = df[bins == b]
            # women at the bottom of each stack, men on top
            sexes = sorted(in_bin["sex"], key=lambda s: s != "female")
            x = edges[b] + BIN_WIDTH / 2
            for level, sex in enumerate(sexes, start=1):
                ax.scatter(x, level, s=220, color=colours[sex], linewidths=0, zorder=3)
            max_stack = max(max_stack, len(sexes))
        n_f = (df["sex"] == "female").sum()
        n_m = (df["sex"] == "male").sum()
        ax.set_title(f"{name}  (n = {len(df)})", loc="left", fontsize=20, pad=34)
        ax.text(0, 1.02, f"{n_f} women · {n_m} men", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=14, color=t["text_secondary"])
        ax.set_xlabel("Age (years)", fontsize=15)
        ax.tick_params(labelsize=13)
        ax.grid(axis="x", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_xlim(*X_LIMITS)
    axes[0].set_ylim(0.3, max_stack + 0.7)
    axes[0].set_yticks(range(1, max_stack + 1))
    axes[0].set_ylabel("People", fontsize=15)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=13, color=c) for c in colours.values()]
    fig.legend(handles, ["Women", "Men"], loc="upper right", ncol=2, fontsize=15,
               bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(w_pad=3)
    return fig


if __name__ == "__main__":
    samples = load_samples()
    for path in save_light_dark(lambda theme: draw(theme, samples), OUT_STEM):
        print(path.relative_to(REPO_ROOT))
