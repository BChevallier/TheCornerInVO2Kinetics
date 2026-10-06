"""Step 5 of the treadmill pipeline: markers of how fast VO2 rises.

For every participant and signal variant, fits phase II by the iterative
fitting window (`shared/phase2_window.py`) and computes the markers defined
in `shared/kinetics_markers.py`: phase II mean response time and
95%-complete time, the slow component (end of trial minus the phase II
asymptote), and the model-free t50 and t90. These are what the piecewise
breakpoint (step 05) is compared against.

Outputs (in `treadmill/results/`):
- `kinetics_markers.csv` — (method, marker) columns
- `phase2_params.csv` — phase II fit (A0, A1, tau1, TD1) and its window end

Run from anywhere: `python treadmill/scripts/08_kinetics_markers.py`.
"""

import pandas as pd

from treadmill_common import MODEL_INPUT_PATHS as INPUT_PATHS
from treadmill_common import RESULTS_DIR
from kinetics_markers import kinetics_marker_tables  # needs treadmill_common imported first (sys.path)


def main():
    data = {method: pd.read_csv(path) for method, path in INPUT_PATHS.items()}
    tables = dict(zip(["kinetics_markers", "phase2_params"], kinetics_marker_tables(data)))
    for name, table in tables.items():
        path = RESULTS_DIR / f"{name}.csv"
        table.to_csv(path)
        print(f"{name} saved to {path}")


if __name__ == "__main__":
    main()
