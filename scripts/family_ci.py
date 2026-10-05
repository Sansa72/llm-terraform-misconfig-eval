#!/usr/bin/env python3
"""
family_ci.py - paired B - A confidence intervals within each
misconfiguration group (codegemma:7b, 100-file test set).

Groups are assigned by the keyword rule from the dataset section: the
groups are tried in a fixed order and a file joins the first group with
a keyword found in its filename.

RUN WITH:
    python family_ci.py

Thesis anchors (files, A%, B%, B-A):
    Encryption/key mgmt   37  51.4  64.9  +13.5
    Residual              33  55.6  57.6   +2.0
    Identity/access       11  51.5  66.7  +15.2
    Logging/monitoring    10  30.0  43.3  +13.3
    Public/network         5  20.0  20.0   +0.0
    Backup/retention       4  25.0  25.0   +0.0
"""

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
}

FILE_COL = "file"
OUTCOME_COL = "outcome"
FIXED_VALUE = "fixed"

# Groups in matching order, as in the dataset section
GROUPS = [
    ("Encryption / key management",
     ["encrypt", "kms", "cmk", "key", "tls", "ssl"]),
    ("Identity / access control",
     ["iam", "policy", "admin", "privilege", "wildcard", "principal"]),
    ("Logging / monitoring",
     ["log", "monitor", "trail", "audit", "alarm"]),
    ("Backup / retention",
     ["backup", "snapshot", "retention", "version"]),
    ("Public exposure / network",
     ["public", "ingress", "egress", "security_group", "acl", "vpc"]),
]


def assign_group(filename):
    name = str(filename).lower()
    for label, kws in GROUPS:
        if any(k in name for k in kws):
            return label
    return "Residual (no keyword match)"


def per_file_rate(names):
    total = None
    for name in names:
        path = os.path.join(BASE, name)
        if not os.path.exists(path):
            sys.exit("Not found: %s" % path)
        df = pd.read_csv(path)
        vals = df[OUTCOME_COL].astype(str).str.strip().str.lower()
        s = pd.Series((vals == FIXED_VALUE).values,
                      index=df[FILE_COL].values).astype(int)
        s = s[~s.index.duplicated(keep="first")]
        total = s if total is None else total.add(s, fill_value=0)
    return (total / len(names) * 100).sort_index()


def main():
    a = per_file_rate(CSVS["A"])
    b = per_file_rate(CSVS["B"])
    groups = pd.Series({f: assign_group(f) for f in a.index})

    order = [g for g, _ in GROUPS] + ["Residual (no keyword match)"]
    print("\n%-30s %5s %6s %6s %7s   %s"
          % ("Group", "files", "A%", "B%", "B-A", "95% CI (pp)"))
    for g in sorted(order, key=lambda x: -np.sum(groups == x)):
        idx = groups[groups == g].index
        n = len(idx)
        if n == 0:
            continue
        xa, xb = a.loc[idx], b.loc[idx]
        d = (xb - xa).values
        m = d.mean()
        if n >= 2 and d.std(ddof=1) > 0:
            se = d.std(ddof=1) / np.sqrt(n)
            tcrit = stats.t.ppf(0.975, n - 1)
            ci = "[%+.1f, %+.1f]" % (m - tcrit * se, m + tcrit * se)
        else:
            ci = "(not computable)"
        print("%-30s %5d %6.1f %6.1f %+7.1f   %s"
              % (g, n, xa.mean(), xb.mean(), m, ci))

    print("\nNote: intervals for groups under ~10 files rest on very few")
    print("files and are wide and unstable by construction.\n")


if __name__ == "__main__":
    main()
