"""The bi-exponential VO2 on-kinetics model: definition and curve-fitting.

Separated from `06_fit_biexponential.py` (which loops the fit over every
participant and file) so the model itself — the part a reader is most
likely to want to check against the paper's methods section — can be read
on its own.

Fitted from 20 s on (phase I excluded) with a multi-start grid; see
`exponential_fitting.py`.

Model form (seven free parameters): a constant baseline plus two
independently-delayed exponential rises, following the two-time-delay
extension to standard mono-exponential VO2 on-kinetics modeling used in
Bell et al. (2001, Exp Physiol 86(5):667-676) for characterizing the
primary/fast phase and the slow-component phase of the response
separately:

    VO2(t) = A0
             + A1 * (1 - exp(-(t - TD1)_clamped / tau1))   # fast component
             + A2 * (1 - exp(-(t - TD2)_clamped / tau2))   # slow component

where `(t - TD)_clamped = clip(t - TD, 0, 300)`, i.e. each component is
exactly zero before its own time delay and stops evolving 300 s after it
starts (well beyond the 240 s trial, so this only guards against overflow
in the optimizer, not a real physiological cutoff).

- `A0`: baseline VO2 before the response begins.
- `A1`, `tau1`, `TD1`: amplitude, time constant, and time delay of the
  fast/primary kinetics component.
- `A2`, `tau2`, `TD2`: amplitude, time constant, and time delay of the
  slower component. `TD2` is the value this project treats as the
  bi-exponential model's "breakpoint", analogous to (but not the same
  quantity as) the piecewise-linear model's breakpoint in
  `05_fit_piecewise.py` - see the top-level README for how the two are
  and aren't comparable.
"""

import numpy as np

from exponential_fitting import multistart_fit, start_grid

PARAM_NAMES = ["A0", "A1", "tau1", "TD1", "A2", "tau2", "TD2"]

# Lower/upper bounds passed to curve_fit, in PARAM_NAMES order. Chosen to
# keep the optimizer within physiologically plausible ranges for a 4-minute
# trial (e.g. TD2 within the trial itself) rather than fit to the data.
# TD2 is otherwise free (Barstow & Mole 1991; Bell et al. 2001): no floor
# is imposed on when the slow component may start.
PARAM_BOUNDS = (
    [0, 0, 1, 0, 0, 10, 30],
    [5, 5, 100, 60, 5, 300, 240],
)

CLAMP_MAX_S = 300

# Multi-start grid (3 x 3 x 3 x 4 = 108 starts); amplitudes are set from the data.
START_GRID = {
    "tau1": [10, 20, 35],
    "TD1": [0, 10, 18],
    "tau2": [30, 100, 250],
    "TD2": [40, 80, 120, 180],
}


def biexponential(t, A0, A1, tau1, TD1, A2, tau2, TD2):
    t = np.asarray(t)
    t1 = np.clip(t - TD1, 0, CLAMP_MAX_S)
    t2 = np.clip(t - TD2, 0, CLAMP_MAX_S)
    fast_component = A1 * (1 - np.exp(-t1 / tau1))
    slow_component = A2 * (1 - np.exp(-t2 / tau2))
    return A0 + fast_component + slow_component


def _starts(y, mono_params=None):
    """Grid starts, plus (if given) the mono-exponential solution with a small
    slow component at several delays, so the bi-exponential can always get
    at least as close as the nested mono-exponential."""
    amplitude = y.max() - y[0]
    yield from start_grid(
        {"A0": y[0], "A1": 0.8 * amplitude, "A2": 0.2 * amplitude}, START_GRID
    )
    if mono_params is not None and not np.isnan(mono_params).any():
        A0, A1, tau1, TD1 = mono_params
        for TD2 in START_GRID["TD2"]:
            yield {"A0": A0, "A1": A1, "tau1": tau1, "TD1": TD1,
                   "A2": 0.01, "tau2": 100, "TD2": TD2}


def fit_biexponential(x, y, y_reference=None, mono_params=None):
    """Fit the bi-exponential model to one participant's VO2 series.

    Fitted to `y` on the fit window (t = 0 plus t >= 20 s; see
    `exponential_fitting.py`), best of the multi-start grid. RMSE% is scored
    against `y_reference` (default: `y` itself) on the same window. Pass the
    unfiltered signal as `y_reference` when `y` is smoothed, so fits to
    differently smoothed versions of the same data are scored against the
    same measurements. `mono_params`, the mono-exponential fit to the same
    series, adds starts next to it (see `_starts`).

    Returns (params, rmse_pct) where `params` is a length-7 array in
    PARAM_NAMES order, or (all-NaN array, NaN) if every start fails.
    """
    y = np.asarray(y, dtype=float)
    if y_reference is None:
        y_reference = y
    return multistart_fit(
        biexponential, PARAM_NAMES, PARAM_BOUNDS, x, y, y_reference,
        _starts(y, mono_params),
    )
