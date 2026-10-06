# Cycling model results

One set of files per time trial (`tt1`, `tt2`). One row per participant; one column per signal variant: `sg` (Savitzky-Golay),
`bw` (Butterworth), `cleaned` (unfiltered). Written by `bike/scripts/05_fit_piecewise.py`, `06_fit_biexponential.py` and `08_kinetics_markers.py`.

- **<trial>_piecewise_breakpoints.csv**
  Breakpoint (s) of the two-segment piecewise-linear fit.

- **<trial>_piecewise_params.csv**
  All piecewise parameters per signal variant: `breakpoint`, and
  `slope1, intercept1, slope2, intercept2` for the two segments (VO2 =
  intercept + slope * t, in L/min and L/min per s). `intercept2` is the
  second line extended back to t = 0. Two header rows (variant, parameter).

- **<trial>_piecewise_rmse_pct.csv**
  RMSE of the piecewise fit against the unfiltered data, as % of its
  amplitude (the same reference for all three signal variants).

- **<trial>_biexponential_breakpoints.csv**
  `TD2`, the slow component's time delay (s), treated as the bi-exponential
  model's breakpoint. Not the same quantity as the piecewise breakpoint; see
  the top-level README.

- **<trial>_biexponential_rmse_pct.csv**
  RMSE of the bi-exponential fit against the unfiltered data, as % of its
  amplitude (the same reference for all three signal variants).

- **<trial>_biexponential_params.csv**
  All seven fitted parameters (`A0, A1, tau1, TD1, A2, tau2, TD2`) per
  signal variant; two header rows (variant, parameter).

- **<trial>_monoexponential_rmse_pct.csv**, **<trial>_monoexponential_params.csv**
  The same for the mono-exponential model: the bi-exponential without its
  slow component (`A0, A1, tau1, TD1`), fitted as the null model.

- **<trial>_exponential_model_selection.csv**
  One row per participant and signal variant: residual sum of squares, AICc
  and BIC of both exponential models, their differences (mono - bi; positive
  favours the bi-exponential), `slow_component_supported` (AICc lower by
  more than 2 for the bi-exponential), and `bi_params_at_bounds`, the
  bi-exponential parameters that ended at a bound. See
  `shared/model_selection.py`.

- **<trial>_biexponential_breakpoints_selected.csv**
  TD2 where the slow component is supported, empty where it is not. Where
  it is not, the fitted TD2 describes a component the data doesn't support.

- **<trial>_kinetics_markers.csv**
  Written by step 08. Per signal variant (two header rows): `mrt` (phase II
  mean response time, TD1 + tau1, s), `fast_end_95` (TD1 + 3 tau1, s),
  `slow_component` (mean of the last 20 s minus the phase II asymptote
  A0 + A1, L/min), `t50`, `t90` (model-free: first time the signal reaches
  50% / 90% of its rise from t = 0 to the mean of the last 20 s, s). See
  `shared/kinetics_markers.py`.

- **<trial>_phase2_params.csv**
  The phase II mono-exponential fit behind those markers (`A0, A1, tau1,
  TD1`) and `window_end` (s), the end of its iterative fitting window; 240
  means the fit never departed from the data. See `shared/phase2_window.py`.

Both exponential models are fitted to t = 0 plus every sample from 20 s on
(phase I excluded), as the best of a grid of starting points; their RMSE%,
RSS and AICc cover those samples only. See `shared/exponential_fitting.py`.
