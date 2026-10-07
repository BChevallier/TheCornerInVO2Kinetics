"""Slide figure: how much smoothing moves the model's tau vs the breakpoint.

Within-participant change (smoothed - unfiltered), in seconds, for phase II
tau (iterative-window mono-exponential, `phase2_params.csv` tau1) and the
piecewise breakpoint, the same quantities analysis/03_smoothing.py tests.
One dot per participant x data set (treadmill, bike tt1, tt2) x filter
(Savitzky-Golay, Butterworth), pooled: the slide's point is the contrast
(tau shifts up, the breakpoint stays on zero), not the per-arm statistics,
which are in analysis/results/smoothing.csv. White bar = median.

Writes presentation/figures/smoothing_shift_{light,dark}.{png,svg}.
Run from anywhere with the project venv.
"""

import numpy as np
import pandas as pd

from usefulness import SETS
from vo2_domains import OUT_DIR, REPO_ROOT, THEMES, plt, save_light_dark

FILTERS = ["sg", "bw"]


def changes():
    tau, bp = [], []
    for folder, prefix in SETS.values():
        p2 = pd.read_csv(folder / f"{prefix}phase2_params.csv", header=[0, 1], index_col=0)
        b = pd.read_csv(folder / f"{prefix}piecewise_breakpoints.csv", index_col=0)
        for f in FILTERS:
            tau.append(p2[f]["tau1"] - p2["cleaned"]["tau1"])
            bp.append(b[f] - b["cleaned"])
    return pd.concat(tau).to_numpy(), pd.concat(bp).to_numpy()


def draw(theme):
    t = THEMES[theme]
    rng = np.random.default_rng(0)  # fixed jitter -> byte-stable figure
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axhline(0, color=t["text_secondary"], lw=1.5, zorder=1)
    for x, values, colour in zip([0, 1], changes(), [t["series"][0], t["series"][1]]):
        jitter = rng.uniform(-0.18, 0.18, len(values))
        ax.plot(x + jitter, values, "o", ms=6, color=colour, alpha=0.6,
                markeredgewidth=0, zorder=2)
        ax.plot([x - 0.26, x + 0.26], [np.median(values)] * 2, color=t["text"],
                lw=3.5, solid_capstyle="round", zorder=3)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1], ["Model (tau)", "Corner"])
    ax.set_ylabel("Change after smoothing (s)", fontsize=15)
    ax.tick_params(axis="x", labelsize=17, length=0)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    for path in save_light_dark(draw, OUT_DIR / "smoothing_shift"):
        print(path.relative_to(REPO_ROOT))
    tau, bp = changes()
    print(f"tau: median {np.median(tau):+.2f} s, range {tau.min():+.1f}..{tau.max():+.1f}")
    print(f"bp:  median {np.median(bp):+.2f} s, range {bp.min():+.1f}..{bp.max():+.1f}")
