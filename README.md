# Piecewise vs. Bi-Exponential

Code and data behind a comparison of two ways to model the VO2 (oxygen
uptake) response during a 4-minute maximal exercise bout:

1. **Piecewise-linear** — a two-segment linear fit with one breakpoint
   (via [`pwlf`](https://github.com/cjekel/piecewise_linear_fit_py)).
2. **Bi-exponential** — a seven-parameter model with two independently
   time-delayed exponential components (fast/primary and slow), fit with
   `scipy.optimize.curve_fit`.

The piecewise breakpoint is compared with standard markers of how fast VO2
rises: the phase II mean response time and 95%-complete time from a
mono-exponential fitted by the iterative fitting window, and the
model-free times to 50% and 90% of the rise.

The question this asks: where does the VO2 response show a "breakpoint" or
inflection, and does the answer depend on which of the two modeling
methods is used, or on which signal-smoothing method is applied
beforehand? Fit quality for both methods is judged by RMSE, expressed as a
percentage of each participant's signal amplitude. Every fit is scored
against the *unfiltered* data, whichever smoothed version it was fitted to:
scoring a fit against the smoothed series itself would make smoothing look
better by construction, since it removes most of the breath-to-breath noise
that the residual consists of.

This repository is not a general-purpose analysis tool — it is a
record of the exact steps used to go from the raw, recorded data to the
figures/tables used in the write-up, kept in a form other researchers can
read and re-run. Every step is a separate, numbered, documented script;
none of the raw data is ever edited by hand, only read and transformed by
these scripts into new files.

## Status

- **Treadmill running** (`treadmill/`) — 4 min at a constant speed, set to
  the speed each runner was predicted to sustain for 4 min. The complete,
  published pipeline: raw data all the way through to breakpoint/model-fit
  results and figures.
- **Cycling** (`bike/`) — a second arm of the same comparison: two self-paced
  4-min cycling time trials (`tt1`, `tt2`) run through the same smoothing and the
  same two models as the treadmill data. Not yet part of the published
  results.

## Repository layout

Each arm is a self-contained folder with the same structure; the code both
arms share lives in `shared/`.

```
shared/                   smoothing and model code used by both arms
  smoothing.py              Savitzky-Golay and Butterworth filters
  piecewise_model.py        two-segment piecewise-linear fit (pwlf)
  exponential_fitting.py    fit window (phase I excluded) and multi-start fitting
  biexponential_model.py    bi-exponential model and fit (curve_fit)
  monoexponential_model.py  the same without the slow component (null model)
  model_selection.py        AICc/BIC comparison of the two exponential models
  phase2_window.py          phase II by the iterative fitting window
  kinetics_markers.py       MRT, 95% time, slow component, t50/t90
  metrics.py                RMSE as % of signal amplitude

treadmill/
  scripts/                steps 01-08, run in order
  data/raw/               as recorded (see its README for columns)
  data/prepared/          outputs of steps 01-04
  results/                outputs of steps 05, 06, 08: breakpoints, RMSE%, parameters, markers
  figures/                output of step 07

bike/
  scripts/                steps 01-08, run in order
  data/raw/               spirometer exports and clock offsets
  data/converted/         exports as per-second CSVs (step 01)
  data/prepared/          outputs of steps 02-04
  data/master_table.csv   per-participant summary table
  results/                outputs of steps 05, 06, 08
  figures/                output of step 07
```

Every data and results folder has a README describing its files.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Every script can be run from any working directory: paths are resolved
relative to the script's own location. Each arm's `*_common.py` holds its
paths and makes `shared/` importable.

## Reproducing the treadmill pipeline

Run the eight numbered scripts in `treadmill/scripts/` in order:

| # | Script | Reads | Writes |
|---|--------|-------|--------|
| 1 | `01_prepare_general_data.py` | `data/raw/general_data.csv` | `data/prepared/general_data.csv` |
| 2 | `02_prepare_time_trial_data.py` | `data/raw/time_trial.csv` | `data/prepared/time_trial.csv` |
| 3 | `03_filter_savitzky_golay.py` | `time_trial.csv` | `data/prepared/time_trial_sg_filtered.csv` |
| 4 | `04_filter_butterworth.py` | `time_trial.csv` | `data/prepared/time_trial_bw_filtered.csv` |
| 5 | `05_fit_piecewise.py` | all three time series | `results/piecewise_*.csv` (breakpoints, parameters, RMSE%) |
| 6 | `06_fit_biexponential.py` | all three time series | `results/biexponential_*.csv`, `results/monoexponential_*.csv`, `results/exponential_model_selection.csv` |
| 7 | `07_plot_participants.py` | filtered data + `general_data.csv` (+ optionally step 5's breakpoints) | `figures/measured_vs_estimated.png` |
| 8 | `08_kinetics_markers.py` | all three time series | `results/kinetics_markers.csv`, `results/phase2_params.csv` |

Paths are relative to `treadmill/`. Steps 3 and 4 are independent of each
other (both only need step 2's output) and can run in either order. Steps 5,
6 and 8 are likewise independent, each comparing its own model across the
same three signal variants: `sg` (Savitzky-Golay filtered), `bw`
(Butterworth filtered), and `cleaned` (no smoothing — the direct output of
step 2).

**Note on reproducibility:** every step is deterministic: re-running
reproduces the committed outputs exactly with the package versions in
`requirements.txt`. The piecewise fit's optimizer (`pwlf`'s differential
evolution) starts from random candidates and is seeded for this reason
(`RANDOM_SEED` in `shared/piecewise_model.py`). With other `numpy`/`scipy`/
`pwlf` versions or BLAS backends, fitted values from steps 5, 6 and 8 can differ
slightly (observed: around the 4th-6th significant digit).

## Reproducing the cycling pipeline

Run the numbered scripts in `bike/scripts/` in order. Each takes an optional
trial argument (`tt1` or `tt2`) and runs both when given none.

| # | Script | Reads | Writes |
|---|--------|-------|--------|
| 1 | `01_xml_to_csv.py` | `data/raw/cpet_exports/<trial>_cpet_pNN.xml` | `data/converted/<trial>/<trial>_pNN.csv` |
| 2 | `02_prepare_time_trial_data.py` | per-second CSVs + `data/raw/delay_anmedu.xlsx` | `data/prepared/<trial>_time_trial.csv` |
| 3 | `03_filter_savitzky_golay.py` | step 2's table | `data/prepared/<trial>_time_trial_sg_filtered.csv` |
| 4 | `04_filter_butterworth.py` | step 2's table | `data/prepared/<trial>_time_trial_bw_filtered.csv` |
| 5 | `05_fit_piecewise.py` | all three time series | `results/<trial>_piecewise_*.csv` (breakpoints, parameters, RMSE%) |
| 6 | `06_fit_biexponential.py` | all three time series | `results/<trial>_biexponential_*.csv`, `results/<trial>_monoexponential_*.csv`, `results/<trial>_exponential_model_selection.csv` |
| 7 | `07_plot_participants.py` | filtered data (+ optionally step 5's breakpoints) | `figures/<trial>_participants.png` |
| 8 | `08_kinetics_markers.py` | all three time series | `results/<trial>_kinetics_markers.csv`, `results/<trial>_phase2_params.csv` |

Paths are relative to `bike/`. Steps 3-8 call the same `shared/` functions
as treadmill steps 3-8, with the same settings, so the two arms differ only
in their data.

Two things differ from the treadmill data and are handled in step 2:

- **Clock offset.** The spirometer's clock lags real time by a
  per-participant amount recorded in `delay_anmedu.xlsx`. The trial starts
  180 s into the session in real time, i.e. at `180 - delay` on the
  spirometer clock.
- **Sampling.** The cycling data is per-second; the treadmill data is in
  5-second samples, and the smoothing settings are defined on that grid.
  Step 2 averages V'O2 into 5-second bins from 0 to 240 s, each labelled by
  its end, so t = 0 holds the last 5 s before the start (the pre-trial
  baseline) as on the treadmill.

Step 2 also leaves out the excluded participants (`EXCLUDED_PARTICIPANTS`
in `bike/scripts/bike_common.py`, with the reason for each). Their raw
exports are kept as recorded.

## Interpreting the two models' "breakpoints"

The piecewise model's breakpoint is the single fitted junction between its
two linear segments. The bi-exponential model's "breakpoint" (as reported
in `biexponential_breakpoints.csv`) is `TD2`: the time delay of its
second (slow) exponential component. These are related but not the same
quantity, and in practice land at quite different points in the response
(the piecewise breakpoint tends to fall much earlier than TD2) — see
`shared/biexponential_model.py`'s module docstring for the full
model definition, and don't assume the two numbers are directly comparable
without accounting for what each one actually measures.

TD2 is only meaningful where the response actually has a slow component. Step
6 therefore also fits the mono-exponential model (the same model without the
slow component) and keeps the bi-exponential only where it lowers AICc by
more than 2 (`shared/model_selection.py`). In many participants it does not:
a 4-minute all-out trial may not give the slow component time to develop.
`biexponential_breakpoints_selected.csv` holds TD2 for the supported cases
only, and `exponential_model_selection.csv` has the full comparison.

Both exponential models are fitted to the baseline sample (t = 0) and the
samples from 20 s on, leaving out phase I (the cardiodynamic phase), which
neither model describes. Each fit is the best of a fixed grid of starting
points, because a single start can end in a local minimum
(`shared/exponential_fitting.py`).

The breakpoint's counterpart in the exponential description is the speed
of the primary (phase II) rise, not TD2. Step 8 fits phase II on its own by
the iterative fitting window: a mono-exponential from 20 s over a window
lengthened from 60 s until the measured data rise consistently above it
(`shared/phase2_window.py` states the rule). Its mean response time
(TD1 + tau1), its 95%-complete time (TD1 + 3 tau1), the slow component
(end of trial minus the phase II asymptote) and the model-free t50/t90
are in `kinetics_markers.csv`.

## License

See [LICENSE](LICENSE).
