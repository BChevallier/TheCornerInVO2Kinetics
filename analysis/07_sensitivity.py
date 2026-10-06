"""Sensitivity of the main results to the phase II fitting window.

The phase II reference (MRT) comes from an iterative fitting window with an
explicit stopping rule (`shared/phase2_window.py`). Here MRT is recomputed
with fixed windows instead — the phase II mono-exponential fitted from 20 s
up to WINDOW_ENDS_S (240 = the whole trial) — and, for each window, the
main results are repeated:

- validity: Pearson r (95% CI) between the piecewise breakpoint and MRT,
  and the median and CV of breakpoint / MRT, per data set;
- reliability (bike tt1 vs tt2): ICC(3,1) with 95% CI, typical error and
  Bland-Altman bias of MRT.

The iterative-window results are included as `window = "iterative"`.

Output: `analysis/results/sensitivity_window.csv`, one row per signal
variant, window, data set and analysis.

Run after both arms' steps 05 and 08: `python analysis/07_sensitivity.py`.
"""

import numpy as np
import pandas as pd

from analysis_common import DATASETS, METHODS, REPO_ROOT, RESULTS_DIR, load_breakpoints, load_two_level
from agreement_stats import bland_altman, icc_3_1_ci, pearson_with_ci, typical_error  # needs analysis_common imported first (sys.path)
from monoexponential_model import fit_monoexponential

WINDOW_ENDS_S = [90, 120, 150, 240]

PREPARED = {
    "treadmill": {
        "cleaned": "treadmill/data/prepared/time_trial.csv",
        "sg": "treadmill/data/prepared/time_trial_sg_filtered.csv",
        "bw": "treadmill/data/prepared/time_trial_bw_filtered.csv",
    },
    **{
        f"bike_{trial}": {
            "cleaned": f"bike/data/prepared/{trial}_time_trial.csv",
            "sg": f"bike/data/prepared/{trial}_time_trial_sg_filtered.csv",
            "bw": f"bike/data/prepared/{trial}_time_trial_bw_filtered.csv",
        }
        for trial in ("tt1", "tt2")
    },
}


def fixed_window_mrt(dataset, method, window_end):
    """MRT (TD1 + tau1) of the mono-exponential fitted up to `window_end`."""
    data = pd.read_csv(REPO_ROOT / PREPARED[dataset][method])
    x = data["time"].values.astype(float)
    within = x <= window_end
    mrt = {}
    for participant in data.columns.drop("time"):
        popt, _ = fit_monoexponential(x[within], data[participant].values[within])
        mrt[participant] = popt[3] + popt[2]
    return pd.Series(mrt)


def all_mrt():
    """(dataset, method, window label) -> per-participant MRT."""
    mrt = {}
    for dataset in DATASETS:
        markers = load_two_level(dataset, "kinetics_markers")
        for method in METHODS:
            mrt[dataset, method, "iterative"] = markers[method]["mrt"]
            for end in WINDOW_ENDS_S:
                mrt[dataset, method, str(end)] = fixed_window_mrt(dataset, method, end)
    return mrt


def sensitivity_rows(mrt):
    windows = ["iterative"] + [str(end) for end in WINDOW_ENDS_S]
    for method in METHODS:
        for window in windows:
            for dataset in DATASETS:
                pair = pd.DataFrame({
                    "bp": load_breakpoints(dataset)[method],
                    "mrt": mrt[dataset, method, window],
                }).dropna()
                r, r_lo, r_hi, _ = pearson_with_ci(pair["mrt"], pair["bp"])
                ratio = pair["bp"] / pair["mrt"]
                yield {
                    "method": method, "window": window, "dataset": dataset,
                    "analysis": "validity", "n": len(pair),
                    "mrt_median": pair["mrt"].median(),
                    "r": r, "r_ci_low": r_lo, "r_ci_high": r_hi,
                    "ratio_median": ratio.median(),
                    "ratio_cv_pct": 100 * ratio.std(ddof=1) / ratio.mean(),
                }
            pair = pd.DataFrame({
                "tt1": mrt["bike_tt1", method, window],
                "tt2": mrt["bike_tt2", method, window],
            }).dropna()
            a, b = pair["tt1"].values, pair["tt2"].values
            icc, icc_lo, icc_hi = icc_3_1_ci(a, b)
            te, te_cv = typical_error(a, b)
            bias, loa_lo, loa_hi = bland_altman(a, b)
            yield {
                "method": method, "window": window, "dataset": "bike_tt1_vs_tt2",
                "analysis": "reliability_mrt", "n": len(pair),
                "icc_3_1": icc, "icc_ci_low": icc_lo, "icc_ci_high": icc_hi,
                "typical_error": te, "typical_error_cv_pct": te_cv,
                "bias": bias, "loa_low": loa_lo, "loa_high": loa_hi,
            }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(sensitivity_rows(all_mrt()))
    path = RESULTS_DIR / "sensitivity_window.csv"
    table.to_csv(path, index=False)
    print(f"sensitivity_window saved to {path}")


if __name__ == "__main__":
    main()
