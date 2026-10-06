"""Markers of how fast VO2 rises, for comparison with the piecewise breakpoint.

From the phase II fit (iterative fitting window, `phase2_window.py`):
- `mrt`: mean response time, TD1 + tau1 (63% of phase II complete).
- `fast_end_95`: TD1 + 3 * tau1, the time phase II is 95% complete.
- `slow_component`: end-of-trial level minus the phase II asymptote
  (A0 + A1), in L/min (as in Rossiter et al. 2001; Murgatroyd et al. 2011).

Model-free, from the signal itself:
- `t50`, `t90`: first time the signal reaches 50% / 90% of the way from the
  baseline (t = 0) to the end-of-trial level (mean of the last
  END_WINDOW_S), linearly interpolated between samples.

In a self-paced trial an end spurt also raises the end-of-trial level.
"""

import numpy as np
import pandas as pd

from monoexponential_model import PARAM_NAMES as MONO_PARAMS
from phase2_window import fit_phase2

END_WINDOW_S = 20

MARKERS = ["mrt", "fast_end_95", "slow_component", "t50", "t90"]
PHASE2_COLUMNS = MONO_PARAMS + ["window_end"]


def window_mean(x, y, end_s):
    """Mean of `y` over the END_WINDOW_S ending at `end_s` (end-labelled bins)."""
    in_window = (x > end_s - END_WINDOW_S) & (x <= end_s)
    return y[in_window].mean()


def time_to_fraction(x, y, fraction):
    """First time `y` reaches `fraction` of its rise from y[0] to the end level."""
    baseline = y[0]
    target = baseline + fraction * (window_mean(x, y, x[-1]) - baseline)
    above = np.nonzero(y >= target)[0]
    if len(above) == 0:
        return np.nan
    i = above[0]
    if i == 0:
        return x[0]
    # linear interpolation between the last sample below and the first above
    return x[i - 1] + (target - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])


def markers(x, y, phase2_params):
    params = dict(zip(MONO_PARAMS, phase2_params))
    return {
        "mrt": params["TD1"] + params["tau1"],
        "fast_end_95": params["TD1"] + 3 * params["tau1"],
        "slow_component": window_mean(x, y, x[-1]) - (params["A0"] + params["A1"]),
        "t50": time_to_fraction(x, y, 0.5),
        "t90": time_to_fraction(x, y, 0.9),
    }


def kinetics_marker_tables(data):
    """Phase II fits and markers for every participant and method.

    `data` maps method ("sg", "bw", "cleaned") to a wide table (`time` plus
    one VO2 column per participant). Everything is computed on each method's
    own signal, except that phase II window departures are always judged
against the unfiltered signal (see `phase2_window.py`). Returns (markers, phase2_params): one row per participant,
    (method, marker) and (method, parameter) columns.
    """
    methods = list(data)
    x = data["cleaned"]["time"].values.astype(float)
    participants = [col for col in data["cleaned"].columns if col != "time"]
    marker_rows, param_rows = {}, {}
    for participant in participants:
        marker_rows[participant], param_rows[participant] = {}, {}
        unfiltered = data["cleaned"][participant].values.astype(float)
        for method in methods:
            y = data[method][participant].values.astype(float)
            popt, window_end = fit_phase2(x, y, unfiltered)
            m = markers(x, y, popt)
            marker_rows[participant] |= {(method, name): m[name] for name in MARKERS}
            values = list(popt) + [window_end]
            param_rows[participant] |= {(method, name): v for name, v in zip(PHASE2_COLUMNS, values)}
    return (
        pd.DataFrame.from_dict(marker_rows, orient="index"),
        pd.DataFrame.from_dict(param_rows, orient="index"),
    )
