"""Slide figures: how VO2 kinetics are usually modelled, and the alternative.

One runner, HA2610: among the cleanest traces (bi-exp RMSE% 1.59, unfiltered)
with a textbook slow component (supported by AICc, TD2 120 s, no parameter at
a bound). ZM311 (the "actually" example) was not used: its fitted TD2 is
60 s, too early to look like the textbook phase III.

Fitted parameters come from the pipeline's results (unfiltered = "cleaned"),
not refitted here. Same frame as the other slides; data and fits scaled by
the participant's highest measured value. Cumulative build-up states, so the
slide can stack them as fragments (each image covers the previous one):

  model_data      dots
  model_biexp     + bi-exponential fit (blue)
  model_phase1-3  + phase brackets and names, one at a time (I, II, III)
  model_piecewise dots + bi-exponential + piecewise fit (orange)
  model_breakpoint  dots + piecewise fit only, the breakpoint circled, arrow
                    and the caption

Phase boundaries follow the model: I = 0..TD1, II = TD1..TD2, III = TD2..end.

Writes presentation/figures/<state>_{light,dark}.{png,svg}.
Run from anywhere with the project venv.
"""

import numpy as np
import pandas as pd

from vo2_domains import OUT_DIR, REPO_ROOT, THEMES, blank_axes, save_light_dark

PID = "HA2610"
DATA = REPO_ROOT / "treadmill/data/prepared/time_trial.csv"
RESULTS = REPO_ROOT / "treadmill/results"
END_S = 240
Y_TOP = 1.3  # room above the data for the phase brackets


def load():
    data = pd.read_csv(DATA)
    scale = data[PID].max()
    bi = pd.read_csv(RESULTS / "biexponential_params.csv", header=[0, 1], index_col=0)
    pw = pd.read_csv(RESULTS / "piecewise_params.csv", header=[0, 1], index_col=0)
    return data["time"], data[PID] / scale, bi.loc[PID, "cleaned"], pw.loc[PID, "cleaned"], scale


def biexponential(t, p):
    fast = p.A1 * (1 - np.exp(-np.clip(t - p.TD1, 0, None) / p.tau1))
    slow = p.A2 * (1 - np.exp(-np.clip(t - p.TD2, 0, None) / p.tau2))
    return p.A0 + fast + slow


def piecewise(t, p):
    return np.where(t < p.breakpoint, p.intercept1 + p.slope1 * t,
                    p.intercept2 + p.slope2 * t)


def draw(theme, layers):
    t_theme = THEMES[theme]
    time_s, vo2, bi, pw, scale = load()
    fig, ax = blank_axes(theme)
    ax.set_ylim(0, Y_TOP)
    ax.plot(time_s / 60, vo2, "o", ms=7, color=t_theme["text_secondary"],
            markeredgewidth=0, alpha=0.75, zorder=2)
    t = np.linspace(0, END_S, 1000)
    if "biexp" in layers:
        ax.plot(t / 60, biexponential(t, bi) / scale, color=t_theme["series"][0],
                lw=3.5, zorder=3)
    if "piecewise" in layers:
        ax.plot(t / 60, piecewise(t, pw) / scale, color=t_theme["series"][1],
                lw=3.5, zorder=4)
    if "breakpoint" in layers:
        # circle the corner of the two lines and point at it from below right
        bp = (pw.breakpoint / 60, float(piecewise(pw.breakpoint, pw)) / scale)
        ax.plot(*bp, "o", ms=44, markerfacecolor="none", markeredgewidth=2.5,
                markeredgecolor=t_theme["text"], zorder=5)
        ax.annotate("But that might be useful", xy=bp, xytext=(1.6, 0.35),
                    fontsize=17, color=t_theme["text"], ha="left", va="center",
                    arrowprops=dict(arrowstyle="-|>", color=t_theme["text"], lw=2,
                                    shrinkA=6, shrinkB=24, mutation_scale=22))
    phases = [("Phase I", 0, bi.TD1), ("Phase II", bi.TD1, bi.TD2), ("Phase III", bi.TD2, END_S)]
    for i, (label, start, end) in enumerate(phases, start=1):
        if f"phase{i}" not in layers:
            continue
        x0, x1, y = start / 60, end / 60, 1.12
        ax.plot([x0, x0, x1, x1], [y - 0.03, y, y, y - 0.03],
                color=t_theme["text_secondary"], lw=1.5)
        if start > 0:
            ax.axvline(x0, ymax=(y - 0.03) / Y_TOP, color=t_theme["muted"], lw=1.2,
                       ls=":", zorder=1)
        ax.text((x0 + x1) / 2, y + 0.03, label, ha="center", va="bottom",
                fontsize=15, color=t_theme["text"])
    fig.tight_layout()
    return fig


STATES = {
    "model_data": [],
    "model_biexp": ["biexp"],
    "model_phase1": ["biexp", "phase1"],
    "model_phase2": ["biexp", "phase1", "phase2"],
    "model_phase3": ["biexp", "phase1", "phase2", "phase3"],
    "model_piecewise": ["biexp", "piecewise"],
    "model_breakpoint": ["piecewise", "breakpoint"],  # blue curve dropped (too busy)
}

if __name__ == "__main__":
    for name, layers in STATES.items():
        for path in save_light_dark(lambda theme: draw(theme, layers), OUT_DIR / name):
            print(path.relative_to(REPO_ROOT))
