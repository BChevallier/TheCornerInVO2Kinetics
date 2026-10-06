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

- **reliability.csv** (`02_reliability.py`)
  Bike tt1 vs tt2, one row per signal variant and quantity: `breakpoint`;
  phase II `mrt`, `tau`, `td`, `fast_end_95`, `slow_component` (step 08);
  `mrt_full_window` (mono-exponential over the whole trial, step 06); `t50`,
  `t90`; `td2_supported` (TD2 where the slow component is supported in both
  trials). Over participants with both values (`n`): trial means;
  `icc_3_1` with 95% CI; `typical_error` (SD of differences / sqrt 2) and
  as % of the mean; Bland-Altman `bias` (tt2 - tt1) with 95% limits of
  agreement (`loa_low`, `loa_high`) and a paired t-test `bias_p`. Typical
  errors in % are not comparable between quantities of different size.

- **reliability_slow_component_verdict.csv** (`02_reliability.py`)
  Agreement of the AICc slow-component verdict between tt1 and tt2 per
  variant: counts, % agreement and Cohen's kappa.

- **smoothing.csv** (`03_smoothing.py`)
  One row per data set, quantity (`breakpoint`, `mrt`, `tau`, `t50`,
  `piecewise_rmse_pct`, `monoexponential_rmse_pct`) and smoothed variant
  (`sg`, `bw`) compared with `cleaned` within participants: medians,
  median and maximum absolute difference, `bias` (variant - cleaned) with
  95% limits of agreement, Wilcoxon signed-rank `wilcoxon_p`, and the
  Friedman test across all three variants (`friedman_p`, repeated on both
  rows of a quantity).

- **simulation_grid.csv**, **simulation_grid_fit.csv**,
  **simulation_slow_component.csv**, **simulation_noise.csv**
  (`04_simulate_exponentials.py`, no inputs) — the piecewise fit applied to
  synthetic mono-exponential responses on the real 5-s grid: noise-free
  over a TD x tau grid (two amplitude pairs), the least-squares fit
  `breakpoint = a + b * TD + c * tau`, the effect of an added slow
  component, and replicates with white noise. Parameters are in the
  script's docstring.

- **sensitivity_window.csv** (`07_sensitivity.py`)
  The main results with phase II MRT from fixed fitting windows (20 s up
  to `window` = 90, 120, 150 or 240 s) instead of the iterative window
  (`window = iterative`). `analysis = validity`: per data set, Pearson `r`
  (95% CI) between breakpoint and MRT, `mrt_median`, `ratio_median` and
  `ratio_cv_pct` of breakpoint / MRT. `analysis = reliability_mrt`
  (`dataset = bike_tt1_vs_tt2`): ICC(3,1) with 95% CI, typical error and
  Bland-Altman bias and limits of agreement of MRT.

- **bike_1hz_participants.csv**, **bike_1hz_summary.csv** (`08_bike_1hz.py`)
  The bike data refitted at 1-s resolution (unfiltered; prepared by
  `bike/scripts/09_prepare_1hz_time_trial.py`). Per participant: breakpoint
  (`bp`) and phase II MRT from fixed windows (`mrt90`, `mrt120`, `mrt150`)
  at 5 s and 1 s. Summary: `agreement` (1 s vs 5 s within participants:
  median and max absolute difference, bias, limits of agreement),
  `validity` (breakpoint vs MRT: r with 95% CI, ratio median and CV) and
  `reliability` (tt1 vs tt2: ICC(3,1), typical error, bias). Bins are
  labelled by their end, so 5-s times sit ~2 s later than 1-s times by
  construction (bin midpoints at T - 2.5 s vs T - 0.5 s).
