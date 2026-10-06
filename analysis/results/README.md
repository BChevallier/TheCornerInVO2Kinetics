# Cross-arm analysis results

Written by the scripts in `analysis/`, from the arms' result files. Data
sets: `treadmill`, `bike_tt1`, `bike_tt2`; signal variants: `cleaned`
(unfiltered), `sg` (Savitzky-Golay), `bw` (Butterworth).

- **validity.csv** (`01_validity.py`)
  One row per data set, variant and marker (`mrt`, `fast_end_95`, `t50`,
  `t90`; see `shared/kinetics_markers.py`): the piecewise breakpoint
  compared with that marker on the same signal variant. `n`; medians of
  both; Pearson `r` with 95% CI (Fisher z) and `r_p`; `spearman_rho`;
  least-squares `breakpoint = intercept + slope * marker` with 95% CIs;
  the ratio breakpoint / marker (`ratio_median`, `ratio_q1`, `ratio_q3`)
  and its coefficient of variation across participants (`ratio_cv_pct`).
