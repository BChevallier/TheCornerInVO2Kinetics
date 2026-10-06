"""Test-retest reliability, bike tt1 vs tt2 (same riders, two time trials).

For every signal variant and quantity — the piecewise breakpoint, the phase
II parameters and markers (MRT, tau, TD, 95% time, slow component), MRT of
the mono-exponential fitted over the whole trial (step 06), the
model-free t50/t90, and TD2 where the slow component is supported in both
trials — reports, over the participants with a value in both trials:

- means in each trial;
- ICC(3,1) with 95% CI (consistency of the participants' ranking);
- typical error (SD of the differences / sqrt 2), in units and as CV%;
- Bland-Altman bias (tt2 - tt1) with 95% limits of agreement, and a paired
  t-test p for the bias.

Also reports how often the AICc slow-component verdict agrees between
trials, with Cohen's kappa.

Outputs (in `analysis/results/`): `reliability.csv` (one row per variant
and quantity), `reliability_slow_component_verdict.csv`.

Run after the bike steps 05, 06 and 08: `python analysis/02_reliability.py`.
"""

import numpy as np
import pandas as pd
from scipy import stats

from analysis_common import METHODS, RESULTS_DIR, load_breakpoints, load_two_level, result_path
from agreement_stats import bland_altman, icc_3_1_ci, typical_error  # needs analysis_common imported first (sys.path)

TRIALS = ("bike_tt1", "bike_tt2")


def quantities(dataset, method):
    """Every quantity compared between trials, as name -> per-participant Series."""
    markers = load_two_level(dataset, "kinetics_markers")[method]
    phase2 = load_two_level(dataset, "phase2_params")[method]
    mono = load_two_level(dataset, "monoexponential_params")[method]
    return {
        "breakpoint": load_breakpoints(dataset)[method],
        "mrt": markers["mrt"],
        "mrt_full_window": mono["TD1"] + mono["tau1"],
        "tau": phase2["tau1"],
        "td": phase2["TD1"],
        "fast_end_95": markers["fast_end_95"],
        "slow_component": markers["slow_component"],
        "t50": markers["t50"],
        "t90": markers["t90"],
        "td2_supported": pd.read_csv(
            result_path(dataset, "biexponential_breakpoints_selected"), index_col=0
        )[method],
    }


def reliability_rows():
    for method in METHODS:
        first, second = (quantities(trial, method) for trial in TRIALS)
        for name in first:
            pair = pd.DataFrame({"tt1": first[name], "tt2": second[name]}).dropna()
            a, b = pair["tt1"].values.astype(float), pair["tt2"].values.astype(float)
            icc, icc_lo, icc_hi = icc_3_1_ci(a, b)
            te, te_cv = typical_error(a, b)
            bias, loa_lo, loa_hi = bland_altman(a, b)
            yield {
                "method": method, "quantity": name, "n": len(pair),
                "mean_tt1": a.mean(), "mean_tt2": b.mean(),
                "icc_3_1": icc, "icc_ci_low": icc_lo, "icc_ci_high": icc_hi,
                "typical_error": te, "typical_error_cv_pct": te_cv,
                "bias": bias, "loa_low": loa_lo, "loa_high": loa_hi,
                "bias_p": stats.ttest_rel(b, a).pvalue,
            }


def verdict_rows():
    for method in METHODS:
        verdicts = []
        for trial in TRIALS:
            selection = pd.read_csv(
                result_path(trial, "exponential_model_selection"), index_col=[0, 1]
            )
            verdicts.append(selection.xs(method, level="method")["slow_component_supported"])
        pair = pd.DataFrame({"tt1": verdicts[0], "tt2": verdicts[1]}).dropna().astype(bool)
        a, b = pair["tt1"], pair["tt2"]
        observed = (a == b).mean()
        expected = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        yield {
            "method": method, "n": len(pair),
            "both_yes": int((a & b).sum()), "both_no": int((~a & ~b).sum()),
            "tt1_only": int((a & ~b).sum()), "tt2_only": int((~a & b).sum()),
            "agreement_pct": 100 * observed,
            "cohen_kappa": (observed - expected) / (1 - expected),
        }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("reliability", reliability_rows()),
        ("reliability_slow_component_verdict", verdict_rows()),
    ]:
        path = RESULTS_DIR / f"{name}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"{name} saved to {path}")


if __name__ == "__main__":
    main()
