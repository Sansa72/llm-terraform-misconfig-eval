#!/usr/bin/env python3

import os
import sys
import json

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except ImportError as e:
    sys.exit("Missing package: %s\n    pip install pandas scipy numpy" % e)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + ""
HUMAN = os.path.join(ROOT, "explanations_test/Human Rating/human_scores.csv")
JUDGE = os.path.join(ROOT, "explanations_test/LLM rating/judge_scores.csv")
REPAIR = os.path.join(ROOT, "data/item_repair_counts.csv")
POSMAP = os.path.join(ROOT, "catch_items/position_map.json")
ITEMMAP = os.path.join(ROOT, "data/item_map.csv")


def main():
    for p in (HUMAN, JUDGE, REPAIR):
        if not os.path.exists(p):
            sys.exit("Not found: %s" % p)

    # ---- ratings, averaged per item per rater
    h = pd.read_csv(HUMAN)
    j = pd.read_csv(JUDGE)
    long = pd.concat([h[["rater", "item_id", "dimension", "score"]],
                      j[["rater", "item_id", "dimension", "score"]]],
                     ignore_index=True)
    avg = (long.groupby(["item_id", "rater"])["score"].mean()
           .unstack("rater"))
    print("\nratings: %d items, raters %s"
          % (len(avg), list(avg.columns)))

    # ---- repair outcomes
    rep = pd.read_csv(REPAIR)
    print("repair file columns:", list(rep.columns))
    key = "item_id" if "item_id" in rep.columns else rep.columns[0]
    rep = rep.set_index(key)
    print("repair rows: %d" % len(rep))

    # ---- keep only the 40 ordinary items
    if os.path.exists(ITEMMAP):
        im = pd.read_csv(ITEMMAP)
        col = "item" if "item" in im.columns else im.columns[0]
        keep = set(str(v).strip() for v in im[col])
        before = len(avg)
        avg = avg[avg.index.astype(str).isin(keep)]
        if len(avg) == 0:                      # labels differ, match on number
            import re
            nums = set(int(re.findall(r"\d+", str(k))[-1]) for k in keep)
            avg = (long.groupby(["item_id", "rater"])["score"].mean()
                   .unstack("rater"))
            avg = avg[[int(re.findall(r"\d+", str(i))[-1]) in nums
                       for i in avg.index]]
        print("items %d -> %d after keeping ordinary items"
              % (before, len(avg)))

    # ---- join
    df = avg.join(rep, how="inner")
    print("joined rows: %d" % len(df))
    if len(df) == 0:
        print("\nCould not join. rating index sample:",
              list(avg.index[:3]))
        print("repair index sample:", list(rep.index[:3]))
        sys.exit("Fix the key names above and run again.")

    b_col = "B_valid" if "B_valid" in df.columns else None
    d_col = "delta" if "delta" in df.columns else None
    if b_col is None or d_col is None:
        sys.exit("Expected B_valid and delta columns; got %s"
                 % list(df.columns))

    raters = [c for c in avg.columns]
    print("\nSpearman correlations with p-values:\n")
    for label, col in [("Valid Condition B repairs (0-3 runs)", b_col),
                       ("Change in valid repairs, A to B", d_col)]:
        print("  %s" % label)
        for r in raters:
            sub = df[[r, col]].dropna()
            rho, p = stats.spearmanr(sub[r], sub[col])
            print("    %-8s rho = %+.2f   p = %.4f   n = %d"
                  % (r, rho, p, len(sub)))
        print("")

    print("Thesis: B repairs  A -0.28  B +0.29  Judge +0.09")
    print("        change     A +0.13  B -0.20  Judge -0.05\n")


if __name__ == "__main__":
    main()
