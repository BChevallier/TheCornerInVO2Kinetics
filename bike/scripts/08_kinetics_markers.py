"""Step 5 of the bike pipeline: markers of how fast VO2 rises.

Same as the treadmill's `08_kinetics_markers.py` (both use
`shared/kinetics_markers.py`): phase II by the iterative fitting window,
its mean response time and 95%-complete time, the slow component, and the
model-free t50 and t90.

Outputs, per trial, in `bike/results/`: `<trial>_kinetics_markers.csv`
and `<trial>_phase2_params.csv` (phase II fit and its window end).

Usage:
    python bike/scripts/08_kinetics_markers.py        # both trials
    python bike/scripts/08_kinetics_markers.py tt1    # one trial
"""

import sys

import pandas as pd

from bike_common import REPO_ROOT, RESULTS_DIR, model_input_paths, requested_trials
from kinetics_markers import kinetics_marker_tables


def run(trial):
    data = {method: pd.read_csv(path) for method, path in model_input_paths(trial).items()}
    tables = dict(zip(["kinetics_markers", "phase2_params"], kinetics_marker_tables(data)))
    for name, table in tables.items():
        path = RESULTS_DIR / f"{trial}_{name}.csv"
        table.to_csv(path)
        print(f"{trial}: {name} saved to {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    for trial in requested_trials(sys.argv):
        run(trial)
