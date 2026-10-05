#!/usr/bin/env python3

import math

Z_ALPHA = 1.959964   # two-sided alpha = 0.05
Z_BETA = 0.841621    # power = 0.80

K0 = 0.40            # null kappa (must be exceeded)
K1 = 0.70            # alternative kappa (expected if rubric works well)

DISTRIBUTIONS = {
    "uniform":         [0.20, 0.20, 0.20, 0.20, 0.20],
    "centre-weighted": [0.10, 0.20, 0.40, 0.20, 0.10],
    "top-skewed":      [0.05, 0.05, 0.15, 0.35, 0.40],  
}


def n_required(p, k0=K0, k1=K1, z_alpha=Z_ALPHA, z_beta=Z_BETA):
    """Planning n to distinguish k1 from k0 given marginal distribution p.

    Uses the simplified variance approximation described in the module
    docstring; not the exact Flack et al. procedure.
    """
    pe = sum(x * x for x in p)                  # chance agreement
    po0 = k0 * (1 - pe) + pe                    # observed agreement under k0
    po1 = k1 * (1 - pe) + pe                    # observed agreement under k1
    v0 = po0 * (1 - po0) / (1 - pe) ** 2        # per-item variance terms
    v1 = po1 * (1 - po1) / (1 - pe) ** 2
    n = (z_alpha * math.sqrt(v0) + z_beta * math.sqrt(v1)) ** 2 / (k1 - k0) ** 2
    return n


if __name__ == "__main__":
    print("Planning approximation (see docstring); "
          f"k0 = {K0}, k1 = {K1}, two-sided alpha = 0.05, power = 0.80\n")
    for label, p in DISTRIBUTIONS.items():
        assert abs(sum(p) - 1.0) < 1e-9, f"{label}: probabilities must sum to 1"
        n = n_required(p)
        pe = sum(x * x for x in p)
        print(f"{label:16s} p = {p}  Pe = {pe:.3f}  "
              f"n = {n:.2f} -> {math.ceil(n)}")
