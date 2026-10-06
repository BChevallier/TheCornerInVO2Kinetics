"""Paths and loaders shared by the cross-arm analysis scripts.

The arm pipelines (`treadmill/scripts/`, `bike/scripts/`) fit the models;
the scripts here compare their outputs. Importing this module puts `shared/`
on `sys.path`, as the arms' `*_common.py` do.
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

ANALYSIS_DIR = REPO_ROOT / "analysis"
RESULTS_DIR = ANALYSIS_DIR / "results"
FIGURES_DIR = ANALYSIS_DIR / "figures"

# Each data set: name -> (results folder, file-name prefix).
DATASETS = {
    "treadmill": (REPO_ROOT / "treadmill" / "results", ""),
    "bike_tt1": (REPO_ROOT / "bike" / "results", "tt1_"),
    "bike_tt2": (REPO_ROOT / "bike" / "results", "tt2_"),
}

METHODS = ["cleaned", "sg", "bw"]


def result_path(dataset, name):
    folder, prefix = DATASETS[dataset]
    return folder / f"{prefix}{name}.csv"


def load_breakpoints(dataset):
    """Piecewise breakpoints: participant x method."""
    return pd.read_csv(result_path(dataset, "piecewise_breakpoints"), index_col=0)


def load_two_level(dataset, name):
    """A (method, quantity) table such as `kinetics_markers`."""
    return pd.read_csv(result_path(dataset, name), header=[0, 1], index_col=0)
