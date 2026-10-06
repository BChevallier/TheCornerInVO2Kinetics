"""Sensitivity to sampling: bike results at 1-s instead of 5-s resolution.

On the 5-s grid the piecewise breakpoint can only move between sample gaps
(see `04_simulate_exponentials.py`). Here the unfiltered bike data at 1-s
resolution (`bike/scripts/09_prepare_1hz_time_trial.py`) are fitted again:

- the piecewise breakpoint (same seeded fit as step 05);
- phase II MRT from mono-exponentials fitted from 20 s to fixed window ends
  (WINDOW_ENDS_S), at both resolutions. The iterative window is not used:
  its stopping rule is defined in 5-s samples.

Outputs (in `analysis/results/`):
- `bike_1hz_participants.csv`: per trial and participant, breakpoint and
  MRT at both resolutions.
- `bike_1hz_summary.csv`: (a) `agreement`: 1 s vs 5 s within participants
  (median and max absolute difference, bias, 95% limits of agreement);
  (b) `validity`: Pearson r (95% CI) and median breakpoint / MRT;
  (c) `reliability`: tt1 vs tt2 ICC(3,1) with 95% CI, typical error, bias.

Run after `bike/scripts/09_prepare_1hz_time_trial.py`:
`python analysis/08_bike_1hz.py`.
"""

import numpy as np
import pandas as pd

from analysis_common import REPO_ROOT, RESULTS_DIR
from agreement_stats import bland_altman, icc_3_1_ci, pearson_with_ci, typical_error  # needs analysis_common imported first (sys.path)
from monoexponential_model import fit_monoexponential
from piecewise_model import fit_piecewise

TRIALS = ("tt1", "tt2")
WINDOW_ENDS_S = [90, 120, 150]
RESOLUTIONS = {"5s": "{trial}_time_trial.csv", "1s": "{trial}_time_trial_1hz.csv"}


def fit_trial(trial):
    rows = {}
    for resolution, name in RESOLUTIONS.items():
        data = pd.read_csv(REPO_ROOT / "bike" / "data" / "prepared" / name.format(trial=trial))
        x = data["time"].values.astype(float)
        for pid in data.columns.drop("time"):
            y = data[pid].values.astype(float)
            row = rows.setdefault(pid, {"trial": trial, "participant": pid})
            row[f"bp_{resolution}"] = fit_piecewise(x, y)[0][0]
            for end in WINDOW_ENDS_S:
                within = x <= end
                popt, _ = fit_monoexponential(x[within], y[within])
                row[f"mrt{end}_{resolution}"] = popt[3] + popt[2]
    return pd.DataFrame(rows.values())


def summary_rows(table):
    quantities = ["bp"] + [f"mrt{end}" for end in WINDOW_ENDS_S]
    for trial, t in table.groupby("trial"):
        for q in quantities:
            a, b = t[f"{q}_5s"].values, t[f"{q}_1s"].values
            bias, lo, hi = bland_altman(a, b)
            yield {"analysis": "agreement", "trial": trial, "quantity": q, "n": len(t),
                   "median_5s": np.median(a), "median_1s": np.median(b),
                   "median_abs_diff": np.median(np.abs(b - a)), "max_abs_diff": np.max(np.abs(b - a)),
                   "bias": bias, "loa_low": lo, "loa_high": hi}
        for resolution in RESOLUTIONS:
            for end in WINDOW_ENDS_S:
                bp, mrt = t[f"bp_{resolution}"], t[f"mrt{end}_{resolution}"]
                r, r_lo, r_hi, _ = pearson_with_ci(mrt, bp)
                ratio = bp / mrt
                yield {"analysis": "validity", "trial": trial, "resolution": resolution,
                       "quantity": f"bp_vs_mrt{end}", "n": len(t), "r": r,
                       "r_ci_low": r_lo, "r_ci_high": r_hi, "ratio_median": ratio.median(),
                       "ratio_cv_pct": 100 * ratio.std(ddof=1) / ratio.mean()}
    wide = table.pivot(index="participant", columns="trial")
    for resolution in RESOLUTIONS:
        for q in quantities:
            col = f"{q}_{resolution}"
            pair = wide[col].dropna()
            a, b = pair["tt1"].values, pair["tt2"].values
            icc, icc_lo, icc_hi = icc_3_1_ci(a, b)
            te, te_cv = typical_error(a, b)
            bias, lo, hi = bland_altman(a, b)
            yield {"analysis": "reliability", "resolution": resolution, "quantity": q,
                   "n": len(pair), "icc_3_1": icc, "icc_ci_low": icc_lo, "icc_ci_high": icc_hi,
                   "typical_error": te, "typical_error_cv_pct": te_cv,
                   "bias": bias, "loa_low": lo, "loa_high": hi}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.concat([fit_trial(trial) for trial in TRIALS], ignore_index=True)
    for name, frame in [("bike_1hz_participants", table),
                        ("bike_1hz_summary", pd.DataFrame(summary_rows(table)))]:
        path = RESULTS_DIR / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"{name} saved to {path}")


if __name__ == "__main__":
    main()
