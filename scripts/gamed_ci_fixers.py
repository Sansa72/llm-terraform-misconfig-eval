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

FIXERS = {
    "codegemma:7b": ("real_test/p7 test", {
        "A": ["test_A_run1.csv", "test_A_run2.csv", "test_A_run3.csv"],
        "B": ["test_B_p7_run1.csv", "test_B_p7_run2.csv",
              "test_B_p7_run3.csv"],
        "C": ["test_C_run1.csv", "test_C_run2.csv", "test_C_run3.csv"],
    }),
    "nemotron-3-super": ("real_test/p7 nemotron test", {
        "A": ["gen_nemotron_A_run1.csv", "gen_nemotron_A_run2.csv",
              "gen_nemotron_A_run3.csv"],
        "B": ["gen_nemotron_B_run1.csv", "gen_nemotron_B_run2.csv",
              "gen_nemotron_B_run3.csv"],
        "C": ["gen_nemotron_C_run1.csv", "gen_nemotron_C_run2.csv",
              "gen_nemotron_C_run3.csv"],
    }),
    "kimi-k2.7-code": ("real_test/p7 kimi test", {
        "A": ["gen_kimi_A_run1.csv", "gen_kimi_A_run2.csv",
              "gen_kimi_A_run3.csv"],
        "B": ["gen_kimi_B_run1.csv", "gen_kimi_B_run2.csv",
              "gen_kimi_B_run3.csv"],
        "C": ["gen_kimi_C_run1.csv", "gen_kimi_C_run2.csv",
              "gen_kimi_C_run3.csv"],
    }),
}

FILE_COL = "file"
OUTCOME_COL = "outcome"
GAMED_VALUE = "gamed_scanner"


def per_file_rate(folder, names, label):
    total = None
    counts = []
    used = 0
    for name in names:
        path = os.path.join(ROOT, folder, name)
        if not os.path.exists(path):
            print("    WARNING: missing %s" % name)
            continue
        df = pd.read_csv(path)
        vals = df[OUTCOME_COL].astype(str).str.strip().str.lower()
        s = pd.Series((vals == GAMED_VALUE).values,
                      index=df[FILE_COL].values).astype(int)
        s = s[~s.index.duplicated(keep="first")]
        counts.append(int(s.sum()))
        total = s if total is None else total.add(s, fill_value=0)
        used += 1
    if total is None:
        sys.exit("No CSVs found for %s" % label)
    pooled = sum(counts)
    cells = len(total) * used
    print("    %s: %d files, gamed per run %s -> pooled %d/%d (%.1f%%)"
          % (label, len(total), counts, pooled, cells, pooled / cells * 100))
    return (total / used * 100).sort_index()


def paired_ci(x, y, label):
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]
    d = (x - y).values
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    t, p = stats.ttest_rel(x.values, y.values)
    star = "" if p >= 0.05 else "   <- interval excludes zero"
    print("    %-32s %+5.1f pp, 95%% CI [%+5.1f, %+5.1f], "
          "t = %5.2f, p = %.4f%s"
          % (label, m, m - tcrit * se, m + tcrit * se, t, p, star))


def main():
    print("")
    results = {}
    for fixer, (folder, csvs) in FIXERS.items():
        print("%s" % fixer)
        rates = {c: per_file_rate(folder, csvs[c], "Condition %s" % c)
                 for c in ("C", "A", "B")}
        results[fixer] = rates
        print("")

    print("Paired differences in invalid scanner pass rate:\n")
    for fixer, rates in results.items():
        print("  %s" % fixer)
        paired_ci(rates["A"], rates["C"], "A - C (finding over file alone)")
        paired_ci(rates["B"], rates["A"], "B - A (explanation over finding)")
        print("")

    print("Thesis pooled rates: codegemma 2.7/4.0/11.3, "
          "nemotron 4.7/5.7/9.3, kimi 4.0/6.0/10.0\n")


if __name__ == "__main__":
    main()
