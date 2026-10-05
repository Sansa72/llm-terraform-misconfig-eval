#!/usr/bin/env python3

import os
import sys

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except ImportError as e:
    sys.exit("Missing package: %s\n    pip install pandas scipy numpy" % e)


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/real_test/p7 test"

CSVS = {
    "A": ["test_A_run1.csv", "test_A_run2.csv", "test_A_run3.csv"],
    "B": ["test_B_p7_run1.csv", "test_B_p7_run2.csv", "test_B_p7_run3.csv"],
    "C": ["test_C_run1.csv", "test_C_run2.csv", "test_C_run3.csv"],
}

FILE_COL = "file"
OUTCOME_COL = "outcome"
FIXED_VALUE = "fixed"


def per_file_rate(names, cond):
    """Per-file percentage of runs in which the file was validly repaired."""
    total = None
    counts = []
    for name in names:
        path = os.path.join(BASE, name)
        if not os.path.exists(path):
            sys.exit("Not found: %s" % path)
        df = pd.read_csv(path)
        flag = (df[OUTCOME_COL].astype(str).str.strip().str.lower()
                == FIXED_VALUE)
        s = pd.Series(flag.values, index=df[FILE_COL].values).astype(int)
        s = s[~s.index.duplicated(keep="first")]
        counts.append(int(s.sum()))
        total = s if total is None else total.add(s, fill_value=0)

    n_runs = len(names)
    rate = (total / n_runs * 100).sort_index()
    pooled = sum(counts)
    cells = len(total) * n_runs
    print("  Condition %s: %d files, fixed per run %s -> pooled %d/%d (%.1f%%)"
          % (cond, len(total), counts, pooled, cells, pooled / cells * 100))
    return rate


def paired_ci(x, y, label):
    """Paired t-test and 95%% CI on the difference x - y."""
    common = x.index.intersection(y.index)
    if len(common) != len(x) or len(common) != len(y):
        print("    (note: comparing %d files common to both)" % len(common))
    x, y = x.loc[common], y.loc[common]
    d = (x - y).values
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    t, p = stats.ttest_rel(x.values, y.values)
    print("  %s : mean %.1f pp, 95%% CI [%.1f, %.1f], t = %.2f, p = %.4f, n = %d"
          % (label, m, m - tcrit * se, m + tcrit * se, t, p, n))


def main():
    print("\nReading from: %s\n" % BASE)
    rates = {}
    for c in ("A", "B", "C"):
        rates[c] = per_file_rate(CSVS[c], c)

    print("\nPaired comparisons:")
    paired_ci(rates["B"], rates["A"], "B - A")
    paired_ci(rates["A"], rates["C"], "A - C")

    print("\nThesis reports:")
    print("  B - A : mean 8.7 pp, 95% CI [2.0, 15.3], t = 2.60, p = 0.0110")
    print("  A - C : mean 28.3 pp, 95% CI [19.9, 36.8]")
    print("\nIf these match, the table is verified.\n")


if __name__ == "__main__":
    main()
