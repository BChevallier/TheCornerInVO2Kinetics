"""Slide figures: what real 4-min bouts look like, in the textbook frame.

One runner and one cyclist, the cleanest of each arm: lowest bi-exponential
RMSE% on the unfiltered signal (treadmill ZM311 1.59%; bike tt2 p32 1.46%,
the lowest of both bike trials). tt1 p19 also ranks high but was flagged as
an artefact in the QC review, so it was not considered. Measured 5-s values
as dots, scaled to the participant's highest value (= 1.0, the level of the
VO2max line in the schematic) because the frame's y axis is unnumbered.
No pre-exercise data: the prepared tables start at the t = 0 baseline.

Writes presentation/figures/example_{runner,cyclist}_{light,dark}.{png,svg}.
Run from anywhere with the project venv.
"""

import pandas as pd

from vo2_domains import OUT_DIR, PAIR_FIGSIZE, REPO_ROOT, THEMES, blank_axes, save_light_dark

EXAMPLES = {
    "runner": (REPO_ROOT / "treadmill/data/prepared/time_trial.csv", "ZM311"),
    "cyclist": (REPO_ROOT / "bike/data/prepared/tt2_time_trial.csv", "p32"),
}


def draw(theme, name):
    path, pid = EXAMPLES[name]
    data = pd.read_csv(path)
    vo2 = data[pid] / data[pid].max()
    fig, ax = blank_axes(theme, PAIR_FIGSIZE)  # shown side by side
    ax.plot(data["time"] / 60, vo2, "o", ms=8, color=THEMES[theme]["series"][0],
            markeredgewidth=0, zorder=3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    for name in EXAMPLES:
        for path in save_light_dark(lambda theme: draw(theme, name), OUT_DIR / f"example_{name}"):
            print(path.relative_to(REPO_ROOT))
