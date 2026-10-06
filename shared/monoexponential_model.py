"""The mono-exponential VO2 on-kinetics model: the bi-exponential without its
slow component.

    VO2(t) = A0 + A1 * (1 - exp(-(t - TD1)_clamped / tau1))

It is exactly `biexponential_model.biexponential` with A2 = 0, and uses the
same bounds for its four parameters, so the two models are nested and can be
compared directly (see `model_selection.py`). It is the reference model for
the primary (fast) phase: at severe intensity the on-transient is typically
mono-exponential (Ozyener et al. 2001). It is also the null model for the
question "does this response have a slow component at all?". Fitted from
20 s on (phase I excluded) with a multi-start grid; see
`exponential_fitting.py`.
"""

import numpy as np

from biexponential_model import CLAMP_MAX_S, PARAM_BOUNDS as BIEXP_BOUNDS
from exponential_fitting import multistart_fit, start_grid

PARAM_NAMES = ["A0", "A1", "tau1", "TD1"]

# The bi-exponential's bounds for the same four parameters.
PARAM_BOUNDS = (
    BIEXP_BOUNDS[0][: len(PARAM_NAMES)],
    BIEXP_BOUNDS[1][: len(PARAM_NAMES)],
)

# Multi-start grid (4 x 3 = 12 starts); A0 and A1 are set from the data.
START_GRID = {
    "tau1": [10, 20, 35, 60],
    "TD1": [0, 8, 16],
}


def monoexponential(t, A0, A1, tau1, TD1):
    t = np.asarray(t)
    t1 = np.clip(t - TD1, 0, CLAMP_MAX_S)
    return A0 + A1 * (1 - np.exp(-t1 / tau1))


def fit_monoexponential(x, y, y_reference=None):
    """Fit the mono-exponential model to one participant's VO2 series.

    Same contract as `biexponential_model.fit_biexponential`: fitted to `y`
    on the fit window (t = 0 plus t >= 20 s), best of a multi-start grid,
    RMSE% scored against `y_reference` (default `y`) on that window.
    Returns (params, rmse_pct), or (all-NaN array, NaN) if every start fails.
    """
    y = np.asarray(y, dtype=float)
    if y_reference is None:
        y_reference = y
    starts = start_grid({"A0": y[0], "A1": y.max() - y[0]}, START_GRID)
    return multistart_fit(
        monoexponential, PARAM_NAMES, PARAM_BOUNDS, x, y, y_reference, starts
    )
