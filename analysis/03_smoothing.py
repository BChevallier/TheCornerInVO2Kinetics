"""Does smoothing the signal change the results?

For every data set and quantity (piecewise breakpoint, phase II MRT and tau,
model-free t50, and the fits' RMSE% against the unfiltered data), compares
the three signal variants within participants:

- Friedman test across `cleaned`, `sg`, `bw`;
- for each smoothed variant vs `cleaned`: median and maximum absolute
  within-participant difference, mean difference (bias) with 95% limits of
  agreement, and a Wilcoxon signed-rank p.

Output: `analysis/results/smoothing.csv`.

Run after both arms' steps 05, 06 and 08: `python analysis/03_smoothing.py`.
"""

import numpy as np
import pandas as pd
from scipy import stats

from analysis_common import DATASETS, RESULTS_DIR, load_breakpoints, load_two_level, result_path
from agreement_stats import bland_altman  # needs analysis_common imported first (sys.path)

VARIANTS = ["cleaned", "sg", "bw"]


def quantities(dataset):
    """name -> participant x variant table."""
    markers = load_two_level(dataset, "kinetics_markers")
    phase2 = load_two_level(dataset, "phase2_params")

    def pick(table, column):
        return pd.DataFrame({v: table[v][column] for v in VARIANTS})

    return {
        "breakpoint": load_breakpoints(dataset)[VARIANTS],
        "mrt": pick(markers, "mrt"),
        "tau": pick(phase2, "tau1"),
        "t50": pick(markers, "t50"),
        "piecewise_rmse_pct": pd.read_csv(
            result_path(dataset, "piecewise_rmse_pct"), index_col=0
        )[VARIANTS],
        "monoexponential_rmse_pct": pd.read_csv(
            result_path(dataset, "monoexponential_rmse_pct"), index_col=0
        )[VARIANTS],
    }


def smoothing_rows(dataset):
    for name, table in quantities(dataset).items():
        table = table.dropna().astype(float)
        friedman_p = stats.friedmanchisquare(*(table[v] for v in VARIANTS)).pvalue
        for variant in ["sg", "bw"]:
            diff = table[variant] - table["cleaned"]
            bias, loa_lo, loa_hi = bland_altman(table["cleaned"], table[variant])
            yield {
                "dataset": dataset, "quantity": name, "variant": variant,
                "n": len(table), "median_cleaned": table["cleaned"].median(),
                "median_variant": table[variant].median(),
                "median_abs_diff": diff.abs().median(), "max_abs_diff": diff.abs().max(),
                "bias": bias, "loa_low": loa_lo, "loa_high": loa_hi,
                "wilcoxon_p": stats.wilcoxon(diff).pvalue if np.any(diff != 0) else np.nan,
                "friedman_p": friedman_p,
            }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([row for d in DATASETS for row in smoothing_rows(d)])
    path = RESULTS_DIR / "smoothing.csv"
    table.to_csv(path, index=False)
    print(f"smoothing saved to {path}")


if __name__ == "__main__":
    main()
