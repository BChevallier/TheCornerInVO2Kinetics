"""Does the piecewise breakpoint track standard markers of VO2 kinetics?

For every data set (treadmill, bike tt1, bike tt2) and signal variant, the
piecewise breakpoint (arm step 05) is compared with the phase II mean
response time, the phase II 95%-complete time and the model-free t50 and
t90 (arm step 08), all estimated on the same signal variant:

- Pearson r with 95% CI, and Spearman rho;
- least-squares breakpoint = intercept + slope * marker, with 95% CIs;
- the ratio breakpoint / marker: median, quartiles, and its coefficient of
  variation across participants (how constant the scaling is).

Output: `analysis/results/validity.csv`, one row per data set, variant and
marker.

Run after both arms' steps 05 and 08: `python analysis/01_validity.py`.
"""

import numpy as np
import pandas as pd
from scipy import stats

from analysis_common import DATASETS, METHODS, RESULTS_DIR, load_breakpoints, load_two_level
from agreement_stats import ols_with_ci, pearson_with_ci  # needs analysis_common imported first (sys.path)

MARKERS = ["mrt", "fast_end_95", "t50", "t90"]


def validity_rows(dataset):
    breakpoints = load_breakpoints(dataset)
    markers = load_two_level(dataset, "kinetics_markers")
    for method in METHODS:
        for marker in MARKERS:
            pair = pd.DataFrame(
                {"bp": breakpoints[method], "marker": markers[method][marker]}
            ).dropna()
            bp, m = pair["bp"].values, pair["marker"].values
            r, r_lo, r_hi, r_p = pearson_with_ci(m, bp)
            slope, slope_lo, slope_hi, icpt, icpt_lo, icpt_hi = ols_with_ci(m, bp)
            ratio = bp / m
            yield {
                "dataset": dataset, "method": method, "marker": marker,
                "n": len(pair),
                "bp_median": np.median(bp), "marker_median": np.median(m),
                "r": r, "r_ci_low": r_lo, "r_ci_high": r_hi, "r_p": r_p,
                "spearman_rho": stats.spearmanr(m, bp).statistic,
                "slope": slope, "slope_ci_low": slope_lo, "slope_ci_high": slope_hi,
                "intercept": icpt, "intercept_ci_low": icpt_lo, "intercept_ci_high": icpt_hi,
                "ratio_median": np.median(ratio),
                "ratio_q1": np.percentile(ratio, 25), "ratio_q3": np.percentile(ratio, 75),
                "ratio_cv_pct": 100 * np.std(ratio, ddof=1) / np.mean(ratio),
            }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([row for d in DATASETS for row in validity_rows(d)])
    path = RESULTS_DIR / "validity.csv"
    table.to_csv(path, index=False)
    print(f"validity saved to {path}")


if __name__ == "__main__":
    main()
