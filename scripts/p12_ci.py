#!/usr/bin/env python3

import os
import sys

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except ImportError as e:
    sys.exit("Missing package: %s\n    pip install pandas scipy numpy" % e)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + ""

P7 = [os.path.join(ROOT, "real_test/p7 test", f) for f in
      ["test_B_p7_run1.csv", "test_B_p7_run2.csv", "test_B_p7_run3.csv"]]
P12 = [os.path.join(ROOT, "real_test/p12 test", f) for f in
       ["test_B_p12_run1.csv", "test_B_p12_run2.csv", "test_B_p12_run3.csv"]]

FILE_COL = "file"
OUTCOME_COL = "outcome"


def per_file_rate(paths, value, label):
    total = None
    counts = []
    for path in paths:
        if not os.path.exists(path):
            sys.exit("Not found: %s" % path)
        df = pd.read_csv(path)
        vals = df[OUTCOME_COL].astype(str).str.strip().str.lower()
        s = pd.Series((vals == value).values,
                      index=df[FILE_COL].values).astype(int)
        s = s[~s.index.duplicated(keep="first")]
        counts.append(int(s.sum()))
        total = s if total is None else total.add(s, fill_value=0)
    pooled = sum(counts)
    cells = len(total) * len(paths)
    print("  %-22s per run %s -> pooled %d/%d (%.1f%%)"
          % (label, counts, pooled, cells, pooled / cells * 100))
    return (total / len(paths) * 100).sort_index()


def paired_ci(x, y, label):
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]
    d = (x - y).values
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    t, p = stats.ttest_rel(x.values, y.values)
    print("  %-22s %+5.1f pp, 95%% CI [%+5.1f, %+5.1f], "
          "t = %5.2f, p = %.4f, n = %d"
          % (label, m, m - tcrit * se, m + tcrit * se, t, p, n))


def main():
    print("\nRepair rate ('fixed'):")
    f7 = per_file_rate(P7, "fixed", "p7")
    f12 = per_file_rate(P12, "fixed", "p12")

    print("\nInvalid scanner pass ('gamed_scanner'):")
    g7 = per_file_rate(P7, "gamed_scanner", "p7")
    g12 = per_file_rate(P12, "gamed_scanner", "p12")

    print("\nPaired differences, p12 - p7:")
    paired_ci(f12, f7, "repair rate")
    paired_ci(g12, g7, "gamed-scanner rate")
    print("")


if __name__ == "__main__":
    main()
