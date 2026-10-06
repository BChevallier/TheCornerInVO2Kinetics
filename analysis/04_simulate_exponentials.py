"""Why does the piecewise breakpoint sit at ~1.6 x the phase II MRT?

Simulation: the 2-segment piecewise-linear model (`shared/piecewise_model.py`,
the same seeded `pwlf` fit the arms use) is fitted to synthetic VO2 responses
sampled like the real data, every 5 s from t = 0 (baseline) to 240 s.

1. Noise-free mono-exponential grid,
       VO2(t) = A0 + A1 * (1 - exp(-(t - TD) / tau))   (A0 before TD),
   for tau in TAU_GRID and TD in TD_GRID, at the reference amplitudes
   (A0, A1) = REFERENCE_AMPLITUDES and at one other pair. A least-squares
   fit is invariant to y -> a + b * y, so the breakpoint should depend on TD
   and tau only; the second pair checks this. On the reference amplitudes,
   bp = a + b * TD + c * tau is fitted by least squares.
2. Slow component: A2 * (1 - exp(-(t - TD2) / TAU2)) added from TD2, for
   a few tau1 values at TD = SLOW_TD; the breakpoint is compared with the
   same response without it (A2 = 0). MRT stays the phase II TD + tau1.
3. Noise: Gaussian white noise of SD NOISE_SD added to the noise-free
   response (all samples, baseline included), N_REPLICATES replicates per
   condition from a fixed seed.

NOISE_SD is the median residual SD of the treadmill phase II fits
(unfiltered signal, inside each participant's phase II window; from
`treadmill/results/phase2_params.csv` and the prepared time-trial data):
0.074 L/min (IQR 0.054-0.087), rounded to 0.075.

Outputs, in `analysis/results/`:
- `simulation_grid.csv`: one row per amplitude pair, TD and tau.
- `simulation_grid_fit.csv`: the bp = a + b * TD + c * tau fit (one row).
- `simulation_slow_component.csv`: one row per tau1, TD2 and A2.
- `simulation_noise.csv`: one row per tau (summary over replicates).

Run: `python analysis/04_simulate_exponentials.py` (no inputs needed).
"""

import time

import numpy as np
import pandas as pd

from analysis_common import RESULTS_DIR
from piecewise_model import fit_piecewise  # needs analysis_common imported first (sys.path)

TIME = np.arange(0, 241, 5, dtype=float)  # s; 5-s samples, t = 0 is baseline

TAU_GRID = np.arange(8, 51, 2, dtype=float)
TD_GRID = [0.0, 5.0, 10.0, 15.0, 20.0]
REFERENCE_AMPLITUDES = (1.0, 3.0)  # (A0, A1), L/min
OTHER_AMPLITUDES = (0.5, 2.0)

SLOW_TAU1 = [15.0, 20.0, 30.0]
SLOW_TD = 10.0
SLOW_A2 = [0.0, 0.1, 0.2, 0.3]  # L/min
SLOW_TD2 = [90.0, 120.0]  # s
TAU2 = 100.0  # s

NOISE_TAU = [15.0, 20.0, 30.0]
NOISE_TD = 10.0
NOISE_SD = 0.075  # L/min; see module docstring
N_REPLICATES = 200
NOISE_SEED = 2024


def response(t, A0, A1, tau, TD, A2=0.0, tau2=TAU2, TD2=np.inf):
    """Mono-exponential (plus optional slow component), A0 before each delay."""
    fast = A1 * (1 - np.exp(-np.clip(t - TD, 0, None) / tau))
    slow = A2 * (1 - np.exp(-np.clip(t - TD2, 0, None) / tau2)) if A2 else 0.0
    return A0 + fast + slow


def breakpoint_of(y):
    params, _ = fit_piecewise(TIME, y)
    return params[0]


def grid_table():
    rows = []
    for A0, A1 in [REFERENCE_AMPLITUDES, OTHER_AMPLITUDES]:
        for TD in TD_GRID:
            for tau in TAU_GRID:
                bp = breakpoint_of(response(TIME, A0, A1, tau, TD))
                mrt = TD + tau
                rows.append({
                    "A0": A0, "A1": A1, "TD": TD, "tau": tau, "mrt": mrt,
                    "breakpoint": bp, "bp_over_mrt": bp / mrt,
                    "bp_minus_td": bp - TD, "bp_minus_td_over_tau": (bp - TD) / tau,
                })
    return pd.DataFrame(rows)


def grid_fit(grid):
    """Least-squares bp = a + b * TD + c * tau on the reference amplitudes."""
    ref = grid[(grid["A0"] == REFERENCE_AMPLITUDES[0]) & (grid["A1"] == REFERENCE_AMPLITUDES[1])]
    X = np.column_stack([np.ones(len(ref)), ref["TD"], ref["tau"]])
    y = ref["breakpoint"].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coef
    r2 = 1 - np.sum(residuals**2) / np.sum((y - y.mean()) ** 2)
    other = grid[(grid["A0"] == OTHER_AMPLITUDES[0]) & (grid["A1"] == OTHER_AMPLITUDES[1])]
    amp_diff = np.abs(ref["breakpoint"].values - other["breakpoint"].values)
    return pd.DataFrame([{
        "n": len(ref), "intercept_a": coef[0], "td_coef_b": coef[1], "tau_coef_c": coef[2],
        "r2": r2, "residual_sd": np.std(residuals, ddof=3),
        "max_abs_residual": np.abs(residuals).max(),
        "amplitude_max_abs_bp_diff": amp_diff.max(),
    }])


def slow_component_table():
    A0, A1 = REFERENCE_AMPLITUDES
    rows = []
    for tau1 in SLOW_TAU1:
        bp_none = breakpoint_of(response(TIME, A0, A1, tau1, SLOW_TD))
        mrt = SLOW_TD + tau1
        for TD2 in SLOW_TD2:
            for A2 in SLOW_A2:
                bp = breakpoint_of(response(TIME, A0, A1, tau1, SLOW_TD, A2=A2, TD2=TD2))
                rows.append({
                    "tau1": tau1, "TD": SLOW_TD, "A2": A2, "TD2": TD2, "tau2": TAU2,
                    "mrt": mrt, "breakpoint": bp, "bp_over_mrt": bp / mrt,
                    "bp_shift": bp - bp_none,
                })
    return pd.DataFrame(rows)


def noise_table():
    A0, A1 = REFERENCE_AMPLITUDES
    rng = np.random.default_rng(NOISE_SEED)
    rows = []
    for tau in NOISE_TAU:
        clean = response(TIME, A0, A1, tau, NOISE_TD)
        bp_clean = breakpoint_of(clean)
        mrt = NOISE_TD + tau
        bps = np.array([
            breakpoint_of(clean + rng.normal(0, NOISE_SD, len(TIME)))
            for _ in range(N_REPLICATES)
        ])
        ratio = bps / mrt
        rows.append({
            "tau": tau, "TD": NOISE_TD, "mrt": mrt, "noise_sd": NOISE_SD,
            "n_replicates": N_REPLICATES, "n_failed": int(np.isnan(bps).sum()),
            "bp_noise_free": bp_clean,
            "bp_mean": np.nanmean(bps), "bp_sd": np.nanstd(bps, ddof=1),
            "bp_q2_5": np.nanpercentile(bps, 2.5), "bp_q97_5": np.nanpercentile(bps, 97.5),
            "bp_over_mrt_mean": np.nanmean(ratio), "bp_over_mrt_sd": np.nanstd(ratio, ddof=1),
        })
    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    grid = grid_table()
    tables = {
        "simulation_grid": grid,
        "simulation_grid_fit": grid_fit(grid),
        "simulation_slow_component": slow_component_table(),
        "simulation_noise": noise_table(),
    }
    for name, table in tables.items():
        path = RESULTS_DIR / f"{name}.csv"
        table.to_csv(path, index=False)
        print(f"{name} saved to {path}")
    print(f"done in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
