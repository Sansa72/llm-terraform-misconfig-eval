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


def per_file_rate(names):
    total = None
    for name in names:
        path = os.path.join(BASE, name)
        if not os.path.exists(path):
            sys.exit("Not found: %s" % path)
        df = pd.read_csv(path)
        flag = (df[OUTCOME_COL].astype(str).str.strip().str.lower()
                == FIXED_VALUE)
        s = pd.Series(flag.values, index=df[FILE_COL].values).astype(int)
        s = s[~s.index.duplicated(keep="first")]
        total = s if total is None else total.add(s, fill_value=0)
    return (total / len(names) * 100).sort_index()


def mean_ci(rate, label):
    n = len(rate)
    m = rate.mean()
    se = rate.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    print("  %s : mean %.1f%%, 95%% CI [%.1f, %.1f], n = %d"
          % (label, m, m - tcrit * se, m + tcrit * se, n))


def main():
    print("\nReading from: %s\n" % BASE)
    print("Per-condition mean repair rate with 95%% interval:")
    for cond in ("C", "A", "B"):
        mean_ci(per_file_rate(CSVS[cond]), "Condition %s" % cond)
    print("\nThesis means: C 19.7, A 48.0, B 56.7."
          "\nThe means must match exactly; the intervals are new.\n")


if __name__ == "__main__":
    main()
