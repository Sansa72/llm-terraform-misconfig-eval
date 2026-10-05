#!/usr/bin/env python3

import sys
import csv
from collections import defaultdict
from statistics import mean, stdev
from math import sqrt

FILE_COL = "file"          # <-- change to your column name
OUTCOME_COL = "outcome"    # <-- change to your column name

FIXED_LABELS = {"fixed", "valid", "valid_repair"}
GAMED_LABELS = {"gamed", "gamed_scanner", "invalid_pass"}

# t critical value, two-sided, 95%, df = 99
T_CRIT_99DF = 1.9842


def load_run(path):
    """Return {file_id: (is_fixed, is_gamed)} for one run."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fid = row[FILE_COL].strip()
            outcome = row[OUTCOME_COL].strip().lower()
            out[fid] = (outcome in FIXED_LABELS, outcome in GAMED_LABELS)
    return out


def per_file_rates(run_paths):
    """Return {file_id: (fixed_rate, gamed_rate)} averaged over the runs."""
    runs = [load_run(p) for p in run_paths]

    files = set(runs[0])
    for r in runs[1:]:
        files &= set(r)

    dropped = set(runs[0]) - files
    if dropped:
        print(f"  note: {len(dropped)} file(s) not present in every run, excluded")

    rates = {}
    for fid in files:
        fixed = [r[fid][0] for r in runs]
        gamed = [r[fid][1] for r in runs]
        rates[fid] = (sum(fixed) / len(fixed), sum(gamed) / len(gamed))
    return rates


def paired_diff(rates_a, rates_b, index):
    """
    Paired difference a - b on one measure (0 = fixed, 1 = gamed),
    in percentage points, with a 95% interval and a t statistic.
    """
    shared = sorted(set(rates_a) & set(rates_b))
    diffs = [(rates_a[f][index] - rates_b[f][index]) * 100 for f in shared]

    n = len(diffs)
    m = mean(diffs)
    sd = stdev(diffs)
    se = sd / sqrt(n)

    half = T_CRIT_99DF * se
    t_stat = m / se if se else float("nan")

    return {
        "n": n,
        "mean": m,
        "low": m - half,
        "high": m + half,
        "t": t_stat,
    }


def fmt(x):
    return f"{x:+.1f}"


def main():
    if len(sys.argv) != 7:
        print(__doc__)
        sys.exit(1)

    p12_paths = sys.argv[1:4]
    p3_paths = sys.argv[4:7]

    print("Loading p12 runs...")
    p12 = per_file_rates(p12_paths)
    print(f"  {len(p12)} files")

    print("Loading p3 runs...")
    p3 = per_file_rates(p3_paths)
    print(f"  {len(p3)} files")

    print()
    print("=" * 62)
    print("Paired comparison, p12 - p3")
    print("=" * 62)

    for label, idx in (("Repair rate", 0), ("Gamed-scanner rate", 1)):
        r = paired_diff(p12, p3, idx)
        print(f"\n{label}")
        print(f"  n            : {r['n']}")
        print(f"  mean diff    : {fmt(r['mean'])} pp")
        print(f"  95% interval : {fmt(r['low'])} to {fmt(r['high'])} pp")
        print(f"  t            : {r['t']:.2f}")
        print(f"  excludes zero: {'yes' if r['low'] > 0 or r['high'] < 0 else 'NO'}")

    print()
    print("=" * 62)
    print("LaTeX rows for tab:p12-diff-ci")
    print("=" * 62)
    rep = paired_diff(p12, p3, 0)
    gam = paired_diff(p12, p3, 1)
    print(f"Repair rate        & ${fmt(rep['mean'])}$ "
          f"& ${fmt(rep['low'])}$ to ${fmt(rep['high'])}$ \\\\")
    print(f"Gamed-scanner rate & ${fmt(gam['mean'])}$ "
          f"& ${fmt(gam['low'])}$ to ${fmt(gam['high'])}$ \\\\")
    print()
    print("Check the sign wording in the prose against these numbers")
    print("before asserting that either interval sits above zero.")


if __name__ == "__main__":
    main()
