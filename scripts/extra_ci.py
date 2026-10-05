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

SETS = {
    # label: (folder, [csv names])
    "codegemma A": ("real_test/p7 test",
                    ["test_A_run1.csv", "test_A_run2.csv",
                     "test_A_run3.csv"]),
    "p7":  ("real_test/p7 test",
            ["test_B_p7_run1.csv", "test_B_p7_run2.csv",
             "test_B_p7_run3.csv"]),
    "p3":  ("data",
            ["test_B_p3_run1.csv", "test_B_p3_run2.csv",
             "test_B_p3_run3.csv"]),
    "p12": ("real_test/p12 test",
            ["test_B_p12_run1.csv", "test_B_p12_run2.csv",
             "test_B_p12_run3.csv"]),
    "nemotron A": ("real_test/p7 nemotron test",
                   ["gen_nemotron_A_run1.csv", "gen_nemotron_A_run2.csv",
                    "gen_nemotron_A_run3.csv"]),
    "nemotron B": ("real_test/p7 nemotron test",
                   ["gen_nemotron_B_run1.csv", "gen_nemotron_B_run2.csv",
                    "gen_nemotron_B_run3.csv"]),
    "nemotron C": ("real_test/p7 nemotron test",
                   ["gen_nemotron_C_run1.csv", "gen_nemotron_C_run2.csv",
                    "gen_nemotron_C_run3.csv"]),
    "kimi A": ("real_test/p7 kimi test",
               ["gen_kimi_A_run1.csv", "gen_kimi_A_run2.csv",
                "gen_kimi_A_run3.csv"]),
    "kimi B": ("real_test/p7 kimi test",
               ["gen_kimi_B_run1.csv", "gen_kimi_B_run2.csv",
                "gen_kimi_B_run3.csv"]),
    "kimi C": ("real_test/p7 kimi test",
               ["gen_kimi_C_run1.csv", "gen_kimi_C_run2.csv",
                "gen_kimi_C_run3.csv"]),
}

FILE_COL = "file"
OUTCOME_COL = "outcome"
FIXED_VALUE = "fixed"


def per_file_rate(label):
    folder, names = SETS[label]
    total = None
    counts = []
    used = 0
    for name in names:
        path = os.path.join(ROOT, folder, name)
        if not os.path.exists(path):
            print("  WARNING: missing %s (using %d run(s) for %s)"
                  % (name, used, label))
            continue
        df = pd.read_csv(path)
        flag = (df[OUTCOME_COL].astype(str).str.strip().str.lower()
                == FIXED_VALUE)
        s = pd.Series(flag.values, index=df[FILE_COL].values).astype(int)
        s = s[~s.index.duplicated(keep="first")]
        counts.append(int(s.sum()))
        total = s if total is None else total.add(s, fill_value=0)
        used += 1
    if total is None:
        sys.exit("No CSVs found for %s" % label)
    rate = (total / used * 100).sort_index()
    print("  %-12s: %d files, fixed per run %s, mean %.1f%%"
          % (label, len(total), counts, rate.mean()))
    return rate


def mean_ci(rate, label):
    n = len(rate)
    m = rate.mean()
    se = rate.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    print("  %-14s: mean %.1f%%, 95%% CI [%.1f, %.1f], n = %d"
          % (label, m, m - tcrit * se, m + tcrit * se, n))


def paired_ci(x, y, label):
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]
    d = (x - y).values
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    t, p = stats.ttest_rel(x.values, y.values)
    print("  %-14s: mean %+.1f pp, 95%% CI [%.1f, %.1f], "
          "t = %.2f, p = %.4f, n = %d"
          % (label, m, m - tcrit * se, m + tcrit * se, t, p, n))


def main():
    print("\nLoading data:")
    r = {}
    for label in SETS:
        r[label] = per_file_rate(label)

    print("\n1a. Prompt means (codegemma:7b):")
    for label in ("p7", "p3", "p12"):
        mean_ci(r[label], label)

    print("\n1b. Prompt differences (paired):")
    paired_ci(r["p12"], r["p7"], "p12 - p7")
    paired_ci(r["p12"], r["p3"], "p12 - p3")
    paired_ci(r["p3"],  r["p7"], "p3 - p7")
    paired_ci(r["p7"],  r["codegemma A"], "p7 - A")
    paired_ci(r["p3"],  r["codegemma A"], "p3 - A")
    paired_ci(r["p12"], r["codegemma A"], "p12 - A")

    print("\n2a. Stronger-fixer condition means:")
    for label in ("nemotron C", "nemotron A", "nemotron B",
                  "kimi C", "kimi A", "kimi B"):
        mean_ci(r[label], label)

    print("\n2b. Stronger-fixer differences (paired, within fixer):")
    paired_ci(r["nemotron B"], r["nemotron A"], "nemotron B - A")
    paired_ci(r["nemotron A"], r["nemotron C"], "nemotron A - C")
    paired_ci(r["kimi B"], r["kimi A"], "kimi B - A")
    paired_ci(r["kimi A"], r["kimi C"], "kimi A - C")

    print("\nThesis means to check against:")
    print("  p7 56.7, p3 57.0, p12 65.0")
    print("  nemotron C 21.0, A 69.3, B 69.7 (B - A = +0.4 by rounded"
          " values, +0.33 unrounded)")
    print("  kimi C 41.7, A 82.3, B 78.7 (B - A = -3.6 by rounded"
          " values, -3.67 unrounded)\n")


if __name__ == "__main__":
    main()
