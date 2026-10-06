"""Bike sensitivity input: the V'O2 table at 1-s resolution instead of 5 s.

Same clock correction and exclusions as `02_prepare_time_trial_data.py`,
but 1-s end-labelled bins from 1 to 240 s (the converted exports have one
row per second; the spirometer holds each breath's value until the next
breath, so neighbouring seconds are not independent). t = 0 is the same
pre-trial baseline as in the 5-s table: the mean over the last 5 s before
the start. Only the unfiltered signal is built: the smoothing settings are
defined on the 5-s grid.

Writes `bike/data/prepared/<trial>_time_trial_1hz.csv` (`time` plus one
column per participant, the participants of the 5-s table). Used by
`analysis/08_bike_1hz.py`.

Usage:
    python bike/scripts/09_prepare_1hz_time_trial.py        # both trials
    python bike/scripts/09_prepare_1hz_time_trial.py tt1    # one trial
"""

import sys

import numpy as np
import pandas as pd

from bike_common import (
    PREPARED_DIR,
    REPO_ROOT,
    TRIAL_DURATION_S,
    converted_path,
    load_delays,
    prepared_path,
    requested_trials,
    seconds_since_trial_start,
)

BASELINE_S = 5
TIMES = np.arange(0, TRIAL_DURATION_S + 1)
VO2_COLUMN = "V'O2"


def per_second_vo2(raw_path, delay_s):
    """V'O2 per 1-s bin (t-1, t], with the 5-s baseline mean at t = 0."""
    df = pd.read_csv(raw_path)
    t = seconds_since_trial_start(pd.to_timedelta(df["t"]), delay_s)
    vo2 = pd.to_numeric(df[VO2_COLUMN], errors="coerce")
    baseline = vo2[(t > -BASELINE_S) & (t <= 0)].mean()
    in_trial = (t > 0) & (t <= TRIAL_DURATION_S)
    per_second = vo2[in_trial].groupby(np.ceil(t[in_trial])).mean()
    return per_second.reindex(TIMES).fillna({0: baseline})


def run(trial):
    delays = load_delays(trial)
    participants = pd.read_csv(prepared_path(trial), nrows=0).columns.drop("time")
    table = pd.DataFrame({"time": TIMES})
    for pid in participants:
        vo2 = per_second_vo2(converted_path(trial, pid), delays[int(pid[1:])])
        missing = vo2.index[vo2.isna()]
        if len(missing):
            print(f"{trial} {pid}: {len(missing)} empty 1-s bins, filled from the previous second")
            vo2 = vo2.ffill()
        table[pid] = vo2.to_numpy()
    path = PREPARED_DIR / f"{trial}_time_trial_1hz.csv"
    table.to_csv(path, index=False)
    print(f"{trial}: {table.shape[1] - 1} participants x {len(table)} s -> {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    for trial in requested_trials(sys.argv):
        run(trial)
