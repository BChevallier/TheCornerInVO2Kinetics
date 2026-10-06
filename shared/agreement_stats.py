"""Statistics for comparing markers: association, test-retest reliability.

Plain numpy/scipy so the repository needs no extra statistics package.
"""

import numpy as np
from scipy import stats


def pearson_with_ci(x, y, confidence=0.95):
    """Pearson r with a Fisher-z confidence interval. Returns (r, lo, hi, p)."""
    r, p = stats.pearsonr(x, y)
    z = np.arctanh(r)
    half = stats.norm.ppf(0.5 + confidence / 2) / np.sqrt(len(x) - 3)
    return r, np.tanh(z - half), np.tanh(z + half), p


def ols_with_ci(x, y, confidence=0.95):
    """Least-squares y = intercept + slope * x with CIs.
    Returns (slope, slope_lo, slope_hi, intercept, intercept_lo, intercept_hi)."""
    fit = stats.linregress(x, y)
    t = stats.t.ppf(0.5 + confidence / 2, len(x) - 2)
    return (
        fit.slope, fit.slope - t * fit.stderr, fit.slope + t * fit.stderr,
        fit.intercept, fit.intercept - t * fit.intercept_stderr,
        fit.intercept + t * fit.intercept_stderr,
    )


def icc_3_1(a, b):
    """ICC(3,1): two-way mixed, consistency, single measurement
    (Shrout & Fleiss 1979; Koo & Li 2016), for two sessions a and b."""
    data = np.column_stack([a, b])
    n, k = data.shape
    grand = data.mean()
    ms_rows = k * np.sum((data.mean(axis=1) - grand) ** 2) / (n - 1)
    residual = data - data.mean(axis=1, keepdims=True) - data.mean(axis=0) + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    return (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)


def icc_3_1_ci(a, b, confidence=0.95):
    """ICC(3,1) with its F-based confidence interval (Shrout & Fleiss 1979)."""
    data = np.column_stack([a, b])
    n, k = data.shape
    grand = data.mean()
    ms_rows = k * np.sum((data.mean(axis=1) - grand) ** 2) / (n - 1)
    residual = data - data.mean(axis=1, keepdims=True) - data.mean(axis=0) + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    f_obs = ms_rows / ms_error
    alpha = 1 - confidence
    df1, df2 = n - 1, (n - 1) * (k - 1)
    f_lo = f_obs / stats.f.ppf(1 - alpha / 2, df1, df2)
    f_hi = f_obs * stats.f.ppf(1 - alpha / 2, df2, df1)
    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)
    return icc, (f_lo - 1) / (f_lo + k - 1), (f_hi - 1) / (f_hi + k - 1)


def typical_error(a, b):
    """Typical error of measurement: SD of the differences / sqrt(2)
    (Hopkins 2000). Also returned as a CV (% of the grand mean)."""
    diff = np.asarray(b) - np.asarray(a)
    te = np.std(diff, ddof=1) / np.sqrt(2)
    return te, 100 * te / np.mean(np.concatenate([a, b]))


def bland_altman(a, b):
    """Mean difference (b - a) and 95% limits of agreement."""
    diff = np.asarray(b) - np.asarray(a)
    bias, sd = diff.mean(), np.std(diff, ddof=1)
    return bias, bias - 1.96 * sd, bias + 1.96 * sd
