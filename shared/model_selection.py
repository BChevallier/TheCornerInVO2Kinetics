"""Is the bi-exponential's slow component supported by the data?

The bi-exponential model can always fit at least as well as the
mono-exponential, since it contains it (A2 = 0). The question is whether the
improvement is worth three extra parameters. Many bi-exponential fits end with
parameters pinned at their bounds (e.g. TD2 at the end of the trial, or A2 at
0), which means the slow component is effectively absent and its TD2 — the
value this project reports as the model's "breakpoint" — is meaningless.

Criterion: small-sample-corrected Akaike information criterion (AICc) for
least-squares fits,

    AIC  = n ln(RSS / n) + 2k
    AICc = AIC + 2k(k + 1) / (n - k - 1)

with k = the model's parameters + 1 (the residual variance). The slow
component is taken as supported when the bi-exponential's AICc is lower than
the mono-exponential's by more than AICC_MARGIN. A difference of 2 or less is
conventionally read as no meaningful evidence either way (Burnham & Anderson,
2002), and the simpler model is then preferred. BIC (`n ln(RSS/n) + k ln n`),
which penalizes parameters more heavily, is reported alongside for
comparison. The same information-criterion approach to choosing between
model families is used by Buchwald & Sveiczer (2006).

n and RSS cover the fit window only (t = 0 plus t >= 20 s; see
`exponential_fitting.py`): phase I is not part of either model.
RSS is always computed against the unfiltered signal, as for RMSE%. For the
smoothed variants the models were fitted to the smoothed series, so the
criteria are an approximation there; the unfiltered ("cleaned") variant is the
one for which they are exact least-squares criteria.
"""

import numpy as np
import pandas as pd

from biexponential_model import PARAM_BOUNDS as BIEXP_BOUNDS
from biexponential_model import PARAM_NAMES as BIEXP_PARAMS
from biexponential_model import biexponential, fit_biexponential
from exponential_fitting import fit_window
from monoexponential_model import PARAM_NAMES as MONO_PARAMS
from monoexponential_model import fit_monoexponential, monoexponential

AICC_MARGIN = 2.0

# A parameter counts as "at a bound" when within this fraction of its allowed
# range of the bound.
BOUND_TOLERANCE = 1e-3


def aicc(rss, n, n_params):
    k = n_params + 1  # + residual variance
    aic = n * np.log(rss / n) + 2 * k
    return aic + 2 * k * (k + 1) / (n - k - 1)


def bic(rss, n, n_params):
    k = n_params + 1
    return n * np.log(rss / n) + k * np.log(n)


def params_at_bounds(popt, bounds, names):
    """Names of the fitted parameters lying on (or within tolerance of) a bound."""
    lower, upper = (np.asarray(b, dtype=float) for b in bounds)
    tolerance = BOUND_TOLERANCE * (upper - lower)
    at_bound = (np.abs(popt - lower) <= tolerance) | (np.abs(popt - upper) <= tolerance)
    return [name for name, flag in zip(names, at_bound) if flag]


def compare_exponential_models(x, y, y_reference):
    """Fit both models to `y` and compare them against `y_reference`.

    Returns a dict with both fits' parameters and RMSE%, their RSS, AICc and
    BIC, the differences (mono - bi; positive favours the bi-exponential),
    whether the slow component is supported, and which bi-exponential
    parameters ended at a bound.
    """
    x = np.asarray(x)
    y_reference = np.asarray(y_reference)
    mask = fit_window(x)
    n = int(mask.sum())

    popt_mono, rmse_mono = fit_monoexponential(x, y, y_reference)
    popt_bi, rmse_bi = fit_biexponential(x, y, y_reference, mono_params=popt_mono)

    result = {
        "popt_bi": popt_bi,
        "popt_mono": popt_mono,
        "rmse_pct_bi": rmse_bi,
        "rmse_pct_mono": rmse_mono,
    }
    if np.isnan(popt_bi).any() or np.isnan(popt_mono).any():
        result.update(slow_component_supported=np.nan, bi_params_at_bounds="")
        return result

    xw, yw = x[mask], y_reference[mask]
    rss_bi = np.sum((yw - biexponential(xw, *popt_bi)) ** 2)
    rss_mono = np.sum((yw - monoexponential(xw, *popt_mono)) ** 2)
    aicc_bi = aicc(rss_bi, n, len(BIEXP_PARAMS))
    aicc_mono = aicc(rss_mono, n, len(MONO_PARAMS))
    bic_bi = bic(rss_bi, n, len(BIEXP_PARAMS))
    bic_mono = bic(rss_mono, n, len(MONO_PARAMS))

    result.update(
        rss_mono=rss_mono,
        rss_bi=rss_bi,
        aicc_mono=aicc_mono,
        aicc_bi=aicc_bi,
        delta_aicc=aicc_mono - aicc_bi,
        bic_mono=bic_mono,
        bic_bi=bic_bi,
        delta_bic=bic_mono - bic_bi,
        slow_component_supported=bool(aicc_mono - aicc_bi > AICC_MARGIN),
        bi_params_at_bounds=" ".join(params_at_bounds(popt_bi, BIEXP_BOUNDS, BIEXP_PARAMS)),
    )
    return result


SELECTION_COLUMNS = [
    "rss_mono", "rss_bi", "aicc_mono", "aicc_bi", "delta_aicc",
    "bic_mono", "bic_bi", "delta_bic",
    "slow_component_supported", "bi_params_at_bounds",
]


def exponential_fit_tables(data):
    """Fit and compare both exponential models for every participant and method.

    `data` maps method ("sg", "bw", "cleaned") to a wide table: `time` plus
    one VO2 column per participant; "cleaned" (unfiltered) is the reference
    every fit is scored against. Returns a dict of output tables, keyed by
    the name each arm's step 06 saves them under:

    - biexponential_breakpoints: TD2 per participant x method
    - biexponential_rmse_pct, monoexponential_rmse_pct
    - biexponential_params, monoexponential_params: (method, param) columns
    - exponential_model_selection: one row per participant and method
    - biexponential_breakpoints_selected: TD2 where the slow component is
      supported, empty where it is not
    """
    methods = list(data)
    participants = [col for col in data["cleaned"].columns if col != "time"]
    x = data["cleaned"]["time"].values

    def per_method_table():
        return pd.DataFrame(index=participants, columns=methods)

    def params_table(names):
        return pd.DataFrame(
            index=participants, columns=pd.MultiIndex.from_product([methods, names])
        )

    tables = {
        "biexponential_breakpoints": per_method_table(),
        "biexponential_rmse_pct": per_method_table(),
        "biexponential_params": params_table(BIEXP_PARAMS),
        "monoexponential_rmse_pct": per_method_table(),
        "monoexponential_params": params_table(MONO_PARAMS),
        "biexponential_breakpoints_selected": per_method_table(),
    }
    selection_rows = []
    td2_index = BIEXP_PARAMS.index("TD2")

    for participant in participants:
        unfiltered = data["cleaned"][participant].values
        for method in methods:
            y = data[method][participant].values
            r = compare_exponential_models(x, y, unfiltered)
            td2 = r["popt_bi"][td2_index]

            tables["biexponential_breakpoints"].loc[participant, method] = td2
            tables["biexponential_rmse_pct"].loc[participant, method] = r["rmse_pct_bi"]
            tables["biexponential_params"].loc[participant, (method,)] = r["popt_bi"]
            tables["monoexponential_rmse_pct"].loc[participant, method] = r["rmse_pct_mono"]
            tables["monoexponential_params"].loc[participant, (method,)] = r["popt_mono"]
            if r["slow_component_supported"] is True:
                tables["biexponential_breakpoints_selected"].loc[participant, method] = td2

            selection_rows.append(
                {"participant": participant, "method": method}
                | {col: r.get(col, np.nan) for col in SELECTION_COLUMNS}
            )

    tables["exponential_model_selection"] = pd.DataFrame(selection_rows).set_index(
        ["participant", "method"]
    )
    return tables
