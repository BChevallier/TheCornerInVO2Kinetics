"""Fit window and multi-start least squares shared by both exponential models.

Fit window: the baseline sample (t = 0) plus every sample from
PHASE_I_CUTOFF_S on. Phase I (the cardiodynamic phase, roughly the first
20 s) is not part of either model, and fitting through it lets a component
describe phase I instead of phase II. Excluding the first ~20 s is standard
practice (Bell et al. 2001; Murias et al. 2011). The t = 0 sample is kept
because without any baseline point the baseline, amplitude and time delay
of the fast component are not jointly identifiable.

Multi-start: `curve_fit` finds a local minimum, and the exponential models
have several. Each fit is started from every point of a fixed grid (plus any
caller-supplied starts) and the lowest residual sum of squares is kept. The
grid is fixed, so results are deterministic.
"""

import itertools

import numpy as np
from scipy.optimize import curve_fit

from metrics import rmse_percent

PHASE_I_CUTOFF_S = 20


def fit_window(x):
    """Boolean mask of the samples the exponential models are fitted to."""
    x = np.asarray(x)
    return (x == 0) | (x >= PHASE_I_CUTOFF_S)


def start_grid(fixed, grids):
    """Starting points: `fixed` (name -> value) combined with every
    combination of `grids` (name -> list of values), as dicts."""
    names = list(grids)
    for values in itertools.product(*(grids[name] for name in names)):
        yield fixed | dict(zip(names, values))


def multistart_fit(model, param_names, bounds, x, y, y_reference, starts):
    """Fit `model` to `y` on the fit window from every start; keep the best.

    `starts` is an iterable of dicts keyed by parameter name; each is clipped
    into `bounds`. The best fit is the one with the lowest RSS against `y`
    (the series being fitted). RMSE% is scored against `y_reference` on the
    fit window. Returns (params, rmse_pct), or (all-NaN, NaN) if every start
    fails.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_reference = np.asarray(y_reference, dtype=float)
    mask = fit_window(x)
    xw, yw = x[mask], y[mask]
    lower, upper = (np.asarray(b, dtype=float) for b in bounds)
    # keep starts strictly inside the bounds (curve_fit rejects p0 on a bound)
    margin = 1e-6 * (upper - lower)

    best_popt, best_rss = None, np.inf
    for start in starts:
        p0 = np.clip([start[name] for name in param_names], lower + margin, upper - margin)
        try:
            popt, _ = curve_fit(model, xw, yw, p0=p0, bounds=(lower, upper), maxfev=10000)
        except (RuntimeError, ValueError):
            continue
        rss = np.sum((yw - model(xw, *popt)) ** 2)
        if rss < best_rss:
            best_popt, best_rss = popt, rss

    if best_popt is None:
        return np.full(len(param_names), np.nan), np.nan
    return best_popt, rmse_percent(y_reference[mask], model(xw, *best_popt))
