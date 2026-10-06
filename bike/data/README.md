# Cycling data

Two 4-min cycling time trials per participant, `tt1` and `tt2`, recorded
breath-by-breath (exported at 1 s) on a Cortex MetaLyzer 3B spirometer.

- **raw/cpet_exports/** — `<trial>_cpet_pNN.xml`, the spirometry software's
  exports (Excel 2003 SpreadsheetML despite the extension). As exported,
  except that the testing facility's contact fields (operator name, named
  contact, phone, email) were blanked before publication.
- **raw/delay_anmedu.xlsx** — per-participant offset (s) by which the
  spirometer clock lags real time, per test. The trial starts 180 s into the
  session in real time, i.e. at `180 - delay` on the spirometer clock.
- **converted/<trial>/** — `<trial>_pNN.csv`, the measurement table of each
  export as CSV, one row per second (`01_xml_to_csv.py`).
- **master_table.csv** — one row per participant: demographics and summary
  performance/physiology for each test (power, heart rate, VO2peak, lactate,
  RPE, efficiency, ...). Not used by the scripts.
- **prepared/** — the model inputs, one wide table per trial: `time` (0-240 s
  in 5-s steps) plus one V'O2 column per participant. Excluded participants
  (`EXCLUDED_PARTICIPANTS` in `scripts/bike_common.py`) are left out here
  and in everything downstream; `raw/` and `converted/` still hold them.
  - `<trial>_time_trial.csv` — 5-s mean V'O2 (`02_prepare_time_trial_data.py`)
  - `<trial>_time_trial_sg_filtered.csv` — Savitzky-Golay smoothed (`03`)
  - `<trial>_time_trial_bw_filtered.csv` — Butterworth smoothed (`04`)
  - `<trial>_time_trial_1hz.csv` — unfiltered V'O2 in 1-s bins, t = 0 the
    5-s pre-trial baseline (`09_prepare_1hz_time_trial.py`); a sampling
    sensitivity input, not fitted by the arm pipeline
