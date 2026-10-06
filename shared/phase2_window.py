"""Phase II (primary component) by the iterative fitting window.

Following Rossiter et al. (2001, 2002) and Murgatroyd et al. (2011): a
mono-exponential is fitted from the end of phase I over a window that starts
at INITIAL_WINDOW_END_S and is lengthened step by step until the fit departs
discernibly and consistently from the data. Phase II is the fit to the
longest window before that departure; the slow component is what the data
add beyond it.

The original papers judge the departure visually. Here it is an explicit
rule, so the result is reproducible. For a window ending at `end`, the
next DEPARTURE_SAMPLES samples after `end` are compared with the fit's
extrapolation. The fit "departs" when those residuals are all positive (the
slow component adds to VO2) and their mean exceeds DEPARTURE_Z standard
errors (the residual SD inside the window / sqrt(DEPARTURE_SAMPLES)).
Residuals are always taken against the unfiltered signal, the measured
values: residuals of a smoothed signal are autocorrelated, so runs of
same-sign residuals would trigger the rule by chance. It must depart for two consecutive
window ends ("consistent"); the window then stops at the first of them. If
it never departs, the window is the whole trial.
"""

import numpy as np

from exponential_fitting import fit_window
from monoexponential_model import PARAM_NAMES, fit_monoexponential, monoexponential

INITIAL_WINDOW_END_S = 60
DEPARTURE_SAMPLES = 3
DEPARTURE_Z = 2.0


def departs(x, y_reference, popt, end):
    """Do the DEPARTURE_SAMPLES samples after `end` depart from the fit?"""
    inside = fit_window(x) & (x <= end)
    after = np.nonzero(x > end)[0][:DEPARTURE_SAMPLES]
    if len(after) < DEPARTURE_SAMPLES:
        return False
    sd = np.std(y_reference[inside] - monoexponential(x[inside], *popt), ddof=len(PARAM_NAMES))
    residuals = y_reference[after] - monoexponential(x[after], *popt)
    return bool(np.all(residuals > 0) and residuals.mean() > DEPARTURE_Z * sd / np.sqrt(DEPARTURE_SAMPLES))


def fit_phase2(x, y, y_reference=None):
    """Iterative-window phase II fit of one VO2 series.

    Fitted to `y`; departures are judged against `y_reference` (default `y`;
    pass the unfiltered signal when `y` is smoothed).

    Returns (params in PARAM_NAMES order, window end in s). Params are NaN if
    a fit fails.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_reference = y if y_reference is None else np.asarray(y_reference, dtype=float)
    ends = [end for end in x if end >= INITIAL_WINDOW_END_S]
    fits = {}
    for end in ends:
        within = x <= end
        fits[end], _ = fit_monoexponential(x[within], y[within])
    departed = {
        end: not np.isnan(fits[end]).any() and departs(x, y_reference, fits[end], end)
        for end in ends
    }
    for end, next_end in zip(ends, ends[1:]):
        if departed[end] and departed[next_end]:
            return fits[end], end
    return fits[ends[-1]], ends[-1]
