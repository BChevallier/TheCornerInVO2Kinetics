"""Slide figures: textbook (schematic, not measured) VO2 response by intensity.

Same axes on every figure so stepping moderate -> heavy -> severe reads as
one curve changing shape. Arbitrary units: baseline 0.2, VO2max 1.0. Curves
are idealised delayed exponentials (+ a delayed slow component above
moderate); no phases are labelled on purpose (introduced later in the talk).
Titles live in the slide HTML, not in the images.

Writes presentation/figures/domain_{empty,empty_narrow,moderate,heavy,severe_max,severe_early}
_{light,dark}.{png,svg}. Run from anywhere with the project venv.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "shared"))
import matplotlib.pyplot as plt  # noqa: E402
from slide_style import THEMES, save_light_dark  # noqa: E402  (black dark theme)

OUT_DIR = REPO_ROOT / "presentation" / "figures"
T_MIN, T_MAX = -1.0, 5.0  # minutes; exercise starts at 0
BASELINE, VO2MAX = 0.2, 1.0
TD = 0.3  # min, delay of the main rise


def rise(t, amplitude, tau, delay):
    """Delayed exponential rise, 0 before `delay`."""
    return np.where(t < delay, 0.0, amplitude * (1 - np.exp(-(t - delay) / tau)))


def drift(t, amplitude, tau, delay):
    """Slow component with a gradual onset (no visible kink on a slide)."""
    x = np.clip(t - delay, 0, None) / tau
    return amplitude * (1 - np.exp(-x ** 2))


# name -> (main amplitude, main tau, slow amplitude, slow delay, slow tau, end time)
CURVES = {
    # back to a flat line within ~2 min
    "moderate": (0.30, 0.4, 0.0, None, None, T_MAX),
    # extra slow drift, levels off late and below VO2max
    "heavy": (0.48, 0.45, 0.14, 1.0, 2.0, T_MAX),
    # slow drift carries VO2 up to VO2max, flat there until the 4-min stop
    "severe_max": (0.60, 0.5, 0.40, 0.8, 2.5, 4.0),
    # slower drift: still rising and below VO2max when it stops
    "severe_early": (0.60, 0.5, 0.40, 0.8, 5.0, 4.0),
}


def curve(name):
    a1, tau1, a2, td2, tau2, end = CURVES[name]
    t = np.linspace(T_MIN, end, 600)
    y = BASELINE + rise(t, a1, tau1, TD)
    if a2:
        y = y + drift(t, a2, tau2, td2)
    return t, np.minimum(y, VO2MAX), end < T_MAX


def blank_axes(theme, figsize=(10, 5.2)):
    """The shared frame: time axis in minutes, unnumbered oxygen-uptake axis.

    Also used as the "empty graph" slide state and by vo2_examples.py, so the
    textbook and the measured curves appear in the same frame.
    """
    t_theme = THEMES[theme]
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, color=t_theme["axis"], lw=1.2, ls=":")
    ax.set_xlim(T_MIN, T_MAX + 0.1)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(range(0, int(T_MAX) + 1))
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("Time (min)", fontsize=16)
    ax.set_ylabel("Oxygen uptake", fontsize=16)
    ax.tick_params(labelsize=14)
    return fig, ax


def draw(theme, name):
    t_theme = THEMES[theme]
    t, y, stops = curve(name)
    # severe variants sit side by side on one slide -> narrower, same text size
    fig, ax = blank_axes(theme, PAIR_FIGSIZE if name.startswith("severe") else (10, 5.2))
    ax.axhline(VO2MAX, color=t_theme["muted"], lw=1.5, ls=(0, (6, 4)))
    ax.text(T_MIN + 0.05, VO2MAX + 0.025, "VO₂max", ha="left", va="bottom",
            fontsize=15, color=t_theme["text_secondary"])
    ax.plot(t, y, color=t_theme["series"][0], lw=4, solid_capstyle="round")
    if stops:
        ax.plot(t[-1], y[-1], marker="X", ms=16, color=t_theme["series"][1],
                markeredgewidth=0, zorder=4)
    fig.tight_layout()
    return fig


PAIR_FIGSIZE = (6.5, 5.2)  # two graphs side by side on one slide: same text size


def draw_empty(theme, figsize=(10, 5.2)):
    fig, _ = blank_axes(theme, figsize)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    for path in save_light_dark(draw_empty, OUT_DIR / "domain_empty"):
        print(path.relative_to(REPO_ROOT))
    for path in save_light_dark(lambda theme: draw_empty(theme, PAIR_FIGSIZE),
                                OUT_DIR / "domain_empty_narrow"):
        print(path.relative_to(REPO_ROOT))
    for name in CURVES:
        for path in save_light_dark(lambda theme: draw(theme, name), OUT_DIR / f"domain_{name}"):
            print(path.relative_to(REPO_ROOT))
