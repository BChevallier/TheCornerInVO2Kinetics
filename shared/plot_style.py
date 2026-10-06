"""Light and dark figure themes; every figure is saved in both.

Figures are drawn by a function that takes a theme and returns a matplotlib
Figure; `save_light_dark` calls it once per theme and writes
`<stem>_light.{png,svg}` and `<stem>_dark.{png,svg}` (the dark version is for
slides with a dark background). Colours are role-based: the series order is
fixed (series 1 is always the first data set or method, etc.) and text never
takes a series colour.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        # categorical, fixed order: blue, orange, aqua, yellow, magenta, green, violet, red
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                   "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "text_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
                   "#d55181", "#008300", "#9085e9", "#e66767"],
    },
}

DPI = 200


def rc_params(theme):
    t = THEMES[theme]
    return {
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["axis"],
        "axes.labelcolor": t["text_secondary"],
        "axes.titlecolor": t["text"],
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=t["series"]),
        "grid.color": t["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "xtick.labelcolor": t["text_secondary"],
        "ytick.labelcolor": t["text_secondary"],
        "text.color": t["text"],
        "legend.frameon": False,
        "legend.labelcolor": t["text_secondary"],
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "font.size": 10,
        "svg.fonttype": "none",
        "svg.hashsalt": "piecewise-vs-biexponential",  # stable element ids
    }


def save_light_dark(draw, stem):
    """Draw and save one figure in both themes.

    `draw(theme)` must build and return a Figure, reading colours from
    `THEMES[theme]` for anything not covered by the rc defaults.
    `stem` is the output path without suffix. Returns the written paths.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for theme in THEMES:
        with plt.rc_context(rc_params(theme)):
            fig = draw(theme)
            for suffix in ("png", "svg"):
                path = stem.parent / f"{stem.name}_{theme}.{suffix}"
                # fixed metadata so re-runs are byte-identical
                metadata = {"Date": None} if suffix == "svg" else {"Software": None}
                fig.savefig(path, dpi=DPI, bbox_inches="tight", metadata=metadata)
                written.append(path)
            plt.close(fig)
    return written
