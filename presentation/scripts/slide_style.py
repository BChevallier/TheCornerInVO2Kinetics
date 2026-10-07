"""shared/plot_style.py with a pure-black dark theme, for the slides only.

The talk uses a true black background (#000), unlike the paper figures'
near-black. Import THEMES / save_light_dark from here instead of plot_style:
the dark theme is replaced in plot_style's own THEMES (rc_params reads it at
call time), in this process only — shared/plot_style.py is not edited.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "shared"))
import plot_style  # noqa: E402

plot_style.THEMES["dark"] = {
    **plot_style.THEMES["dark"],
    "surface": "#000000",
    "grid": "#262625",
    "axis": "#3a3a37",
}

THEMES = plot_style.THEMES
save_light_dark = plot_style.save_light_dark
