"""Slide figures: why the breakpoint might be useful.

useful_tracks  breakpoint vs phase II MRT (= TD1 + tau1 from the iterative-
               window mono-exponential, `phase2_params.csv`), unfiltered
               signal, every runner and both bike trials, with the line
               breakpoint = 1.6 x MRT. Same data as analysis fig2 (validity:
               r 0.88 / 0.91 / 0.80); lay axis labels.
useful_repeat  cyclists' trial 1 -> trial 2, one line per person, MRT (left)
               and breakpoint (right). Both panels span the same number of
               seconds, so line steepness compares directly (typical error
               3.3 s vs 3.5 s, analysis/results/reliability.csv).

Writes presentation/figures/useful_{tracks,repeat}_{light,dark}.{png,svg}.
Run from anywhere with the project venv.
"""

import numpy as np
import pandas as pd

from vo2_domains import OUT_DIR, REPO_ROOT, THEMES, plt, save_light_dark

RATIO = 1.6
SPAN_S = 45  # y range of both repeatability panels, seconds
SETS = {
    "treadmill": (REPO_ROOT / "treadmill/results", ""),
    "tt1": (REPO_ROOT / "bike/results", "tt1_"),
    "tt2": (REPO_ROOT / "bike/results", "tt2_"),
}


def load(name):
    folder, prefix = SETS[name]
    bp = pd.read_csv(folder / f"{prefix}piecewise_breakpoints.csv", index_col=0)["cleaned"]
    p2 = pd.read_csv(folder / f"{prefix}phase2_params.csv", header=[0, 1], index_col=0)["cleaned"]
    return pd.DataFrame({"bp": bp, "mrt": p2["TD1"] + p2["tau1"]})


def draw_tracks(theme):
    t = THEMES[theme]
    runners = load("treadmill")
    cyclists = pd.concat([load("tt1"), load("tt2")])
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.array([15, 50])
    ax.plot(x, RATIO * x, color=t["muted"], lw=1.8, ls=(0, (6, 4)), zorder=1)
    ax.text(x[1], RATIO * x[1] + 1.5, f"corner = {RATIO} × model", ha="right", va="bottom",
            fontsize=14, color=t["text_secondary"])
    for df, colour, label in [(runners, t["series"][0], "Runners"),
                              (cyclists, t["series"][3], "Cyclists (both trials)")]:
        ax.plot(df["mrt"], df["bp"], "o", ms=9, color=colour, alpha=0.85,
                markeredgewidth=0, label=label, zorder=3)
    ax.set_xlim(*x)
    ax.set_ylim(25, 85)
    ax.set_xlabel("Model: average duration of the main rise (s)", fontsize=15)
    ax.set_ylabel("Corner of the two lines (s)", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=14, loc="upper left", handletextpad=0.3)
    fig.tight_layout()
    return fig


def draw_repeat(theme):
    t = THEMES[theme]
    tt1, tt2 = load("tt1"), load("tt2")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6))
    panels = [("mrt", "Model", t["series"][0]), ("bp", "Corner", t["series"][1])]
    for ax, (col, title, colour) in zip(axes, panels):
        for pid in tt1.index:
            ax.plot([0, 1], [tt1.at[pid, col], tt2.at[pid, col]], "-o", color=colour,
                    lw=1.6, ms=6, alpha=0.7, markeredgewidth=0)
        centre = np.median(np.r_[tt1[col], tt2[col]])
        ax.set_ylim(centre - SPAN_S / 2, centre + SPAN_S / 2)
        ax.set_xlim(-0.25, 1.25)
        ax.set_xticks([0, 1], ["Trial 1", "Trial 2"])
        ax.set_title(title, fontsize=18, pad=10)
        ax.tick_params(labelsize=13)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Seconds", fontsize=15)
    fig.tight_layout(w_pad=4)
    return fig


if __name__ == "__main__":
    for name, draw in [("useful_tracks", draw_tracks), ("useful_repeat", draw_repeat)]:
        for path in save_light_dark(draw, OUT_DIR / name):
            print(path.relative_to(REPO_ROOT))
    # cross-check against the analysis results the slides quote
    for name in SETS:
        df = load(name)
        print(name, "r =", round(df["bp"].corr(df["mrt"]), 3),
              "median ratio =", round((df["bp"] / df["mrt"]).median(), 2))
