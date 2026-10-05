#!/usr/bin/env python3

import os
import re
import sys
import json
import itertools

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except ImportError as e:
    sys.exit("Missing package: %s\n    pip install pandas scipy numpy" % e)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + ""
HUMAN = os.path.join(ROOT, "explanations_test/Human Rating/human_scores.csv")
JUDGE = os.path.join(ROOT, "explanations_test/LLM rating/judge_scores.csv")
POSMAP = os.path.join(ROOT, "catch_items/position_map.json")
ITEMMAP = os.path.join(ROOT, "data/item_map.csv")

CATCH_FALLBACK = [10, 14, 27, 29, 44]   # used only if position_map.json fails
N_BOOT = 10000
SEED = 42


# ---------------------------------------------------------------- stats
def fleiss_kappa(counts):
    """counts: rows = subjects, cols = categories, cells = n raters."""
    counts = np.asarray(counts, float)
    n_sub, _ = counts.shape
    per = counts.sum(axis=1)
    n = per[0]
    if not np.allclose(per, n):
        raise ValueError("unequal raters per subject")
    p_j = counts.sum(axis=0) / (n_sub * n)
    P_i = ((counts ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_e = (p_j ** 2).sum()
    return (P_i.mean() - P_e) / (1 - P_e) if (1 - P_e) else np.nan


def weighted_kappa(x, y, weights="linear", k_min=1, k_max=5):
    """Cohen's weighted kappa on an ordinal scale."""
    x = np.asarray(x, int); y = np.asarray(y, int)
    cats = np.arange(k_min, k_max + 1); k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    O = np.zeros((k, k))
    for a, b in zip(x, y):
        O[idx[a], idx[b]] += 1
    O /= O.sum()
    hx = np.array([(x == c).sum() for c in cats], float) / len(x)
    hy = np.array([(y == c).sum() for c in cats], float) / len(y)
    E = np.outer(hx, hy)
    i, j = np.indices((k, k))
    W = np.abs(i - j) / (k - 1) if weights == "linear" \
        else ((i - j) ** 2) / ((k - 1) ** 2)
    den = (W * E).sum()
    return 1 - (W * O).sum() / den if den else np.nan


def fleiss_from_frame(d, raters):
    counts = np.zeros((len(d), 5))
    for r in raters:
        for i, v in enumerate(d[r].values):
            counts[i, int(v) - 1] += 1
    return fleiss_kappa(counts)


def bootstrap_ci(d, fn, item_col="item_id", n_boot=N_BOOT, seed=SEED):
    """Cluster bootstrap: resample whole explanations, not single ratings."""
    rng = np.random.default_rng(seed)
    items = d[item_col].unique()
    groups = {i: d[d[item_col] == i] for i in items}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(items, len(items), replace=True)
        boot = pd.concat([groups[i] for i in pick], ignore_index=True)
        try:
            v = fn(boot)
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    if not vals:
        return np.nan, np.nan
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


# ----------------------------------------------------------------- main
def main():
    for p in (HUMAN, JUDGE):
        if not os.path.exists(p):
            sys.exit("Not found: %s" % p)

    h = pd.read_csv(HUMAN)
    j = pd.read_csv(JUDGE)
    long = pd.concat([h[["rater", "item_id", "dimension", "score"]],
                      j[["rater", "item_id", "dimension", "score"]]],
                     ignore_index=True)

    print("\nLoaded %d human rows + %d judge rows = %d"
          % (len(h), len(j), len(long)))
    print("raters in data :", sorted(long["rater"].unique()))
    print("dimensions     :", sorted(long["dimension"].unique()))
    print("items          :", long["item_id"].nunique())

    # ---- work out which items are the 40 ordinary ones
    keep = None
    if os.path.exists(ITEMMAP):
        im = pd.read_csv(ITEMMAP)
        col = "item" if "item" in im.columns else im.columns[0]
        keep = set(str(v).strip() for v in im[col])
        print("ordinary items : %d from item_map.csv" % len(keep))

    catch_nums = set()
    if os.path.exists(POSMAP):
        try:
            pm = json.load(open(POSMAP))
            mapping = pm.get("mapping", pm if isinstance(pm, list) else [])
            for m in mapping:
                if m.get("is_catch"):
                    d = re.findall(r"\d+", str(m.get("source", "")))
                    if d:
                        catch_nums.add(int(d[-1]))
            print("catch item nos : %s (from position_map.json)"
                  % sorted(catch_nums))
        except Exception as e:
            print("(could not parse position_map.json: %s)" % e)

    # ---- pivot to one row per (item, dimension)
    wide = long.pivot_table(index=["item_id", "dimension"],
                            columns="rater", values="score").reset_index()
    wide.columns.name = None
    raters = [c for c in wide.columns if c not in ("item_id", "dimension")]

    before = wide["item_id"].nunique()
    def item_no(v):
        d = re.findall(r"\d+", str(v))
        return int(d[-1]) if d else -1

    if keep:
        mask = wide["item_id"].astype(str).str.strip().isin(keep)
        if mask.sum() == 0:                      # labels differ, match on number
            keep_nums = set(item_no(k) for k in keep)
            mask = wide["item_id"].map(item_no).isin(keep_nums)
        wide = wide[mask]
    elif catch_nums:
        wide = wide[~wide["item_id"].map(item_no).isin(catch_nums)]
    wide = wide.dropna(subset=raters)
    wide[raters] = wide[raters].astype(int)
    print("items %d -> %d after dropping catch items; %d rating rows\n"
          % (before, wide["item_id"].nunique(), len(wide)))

    if wide["item_id"].nunique() != 40:
        print("WARNING: expected 40 ordinary items, got %d. Check the catch "
              "list above.\n" % wide["item_id"].nunique())

    # ---- Fleiss
    k = fleiss_from_frame(wide, raters)
    lo, hi = bootstrap_ci(wide, lambda d: fleiss_from_frame(d, raters))
    print("Fleiss' kappa (three evaluators, unweighted)")
    print("   computed : %+.3f   95%% CI [%.3f, %.3f]" % (k, lo, hi))
    print("   thesis   : -0.047   95%% CI [-0.122, 0.023]\n")

    # ---- pairwise
    for wt in ("linear", "quadratic"):
        print("Pairwise weighted kappa  [%s weights]" % wt)
        for r1, r2 in itertools.combinations(raters, 2):
            x, y = wide[r1].values, wide[r2].values
            kw = weighted_kappa(x, y, weights=wt)
            lo, hi = bootstrap_ci(
                wide, lambda d, a=r1, b=r2, w=wt:
                weighted_kappa(d[a].values, d[b].values, weights=w))
            same = np.mean(x == y) * 100
            within = np.mean(np.abs(x - y) <= 1) * 100
            m = wide.groupby("item_id")[[r1, r2]].mean()
            rho, _ = stats.spearmanr(m[r1], m[r2])
            print("   %-16s kappa %+.3f  CI [%+.3f, %+.3f]  same %2.0f%%  "
                  "within-1 %2.0f%%  rho %+.2f"
                  % ("%s & %s" % (r1, r2), kw, lo, hi, same, within, rho))
        print()

    print("Thesis values to compare against:")
    print("   A & B      0.013  CI [-0.083, 0.120]  same 42%  "
          "within-1 79%  rho -0.33")
    print("   A & Judge  0.064  CI [-0.037, 0.166]  same 32%  "
          "within-1 94%  rho -0.15")
    print("   B & Judge  0.192  CI [ 0.090, 0.316]  same 34%  "
          "within-1 79%  rho +0.39\n")


if __name__ == "__main__":
    main()
