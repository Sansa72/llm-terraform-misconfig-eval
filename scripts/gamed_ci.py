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
GAMED_VALUE = "gamed_scanner"


def per_file_rate(names, cond):
    """Per-file percentage of runs which produced an invalid scanner pass."""
    total = None
    counts = []
    for name in names:
        path = os.path.join(BASE, name)
        if not os.path.exists(path):
            sys.exit("Not found: %s" % path)
        df = pd.read_csv(path)
        vals = df[OUTCOME_COL].astype(str).str.strip().str.lower()
        flag = vals == GAMED_VALUE
        if flag.sum() == 0:
            print("  NOTE: no '%s' rows in %s; values present: %s"
                  % (GAMED_VALUE, name, sorted(vals.unique())))
        s = pd.Series(flag.values, index=df[FILE_COL].values).astype(int)
        s = s[~s.index.duplicated(keep="first")]
        counts.append(int(s.sum()))
        total = s if total is None else total.add(s, fill_value=0)

    n_runs = len(names)
    pooled = sum(counts)
    cells = len(total) * n_runs
    print("  Condition %s: %d files, gamed per run %s -> pooled %d/%d (%.1f%%)"
          % (cond, len(total), counts, pooled, cells, pooled / cells * 100))
    return (total / n_runs * 100).sort_index()


def mean_ci(rate, label):
    n = len(rate)
    m = rate.mean()
    se = rate.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    print("  %-28s mean %5.1f%%, 95%% CI [%.1f, %.1f]"
          % (label, m, m - tcrit * se, m + tcrit * se))


def paired_ci(x, y, label):
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]
    d = (x - y).values
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    t, p = stats.ttest_rel(x.values, y.values)
    print("  %-28s mean %+5.1f pp, 95%% CI [%+.1f, %+.1f], "
          "t = %5.2f, p = %.4f, n = %d"
          % (label, m, m - tcrit * se, m + tcrit * se, t, p, n))


def main():
    print("\nReading from: %s\n" % BASE)
    rates = {}
    for c in ("C", "A", "B"):
        rates[c] = per_file_rate(CSVS[c], c)

    print("\nPer-condition invalid scanner pass rate:")
    mean_ci(rates["C"], "C (file alone)")
    mean_ci(rates["A"], "A (scanner finding)")
    mean_ci(rates["B"], "B (explanation)")

    print("\nPaired differences:")
    paired_ci(rates["A"], rates["C"], "A - C (finding over file)")
    paired_ci(rates["B"], rates["A"], "B - A (explanation over finding)")

    print("\nThesis pooled rates: C 2.7%, A 4.0%, B 11.3%\n")


if __name__ == "__main__":
    main()
