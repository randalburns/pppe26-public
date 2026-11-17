#!/usr/bin/env python3
"""
Plot the matrix-multiply kernels from mm_bench.cpp on the single-core
roofline, one plot per matrix size. Reuses roofline.py from the roofline
examples for the ceilings and label placement.

Usage: python3 plot_mm.py <results.csv> <ceilings.txt> [--machine NAME]

results.csv   RESULT,<build>,<n>,<kernel>,<ms>,<GFLOP/s> lines from mm_bench
ceilings.txt  output of roofline_example's stream and peak_flops
              (PEAK_BANDWIDTH=, PEAK_GFLOPS_SIMD_FMA=, ...)

Writes mm_roofline_<n>.png for each n in the results.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "roofline", "roofline_example"))
import roofline  # noqa: E402


# The blog's sequence, in order: (build, kernel prefix, label). For the two
# tiled kernels, the best tile size measured is plotted and named.
STEPS = [
    ("O0",   "ijk",       "ijk, no flags"),
    ("O3",   "ijk",       "ijk, -O3 -march=native"),
    ("fast", "ijk",       "ijk, + -ffast-math"),
    ("fast", "ijk_local", "ijk, local accumulator"),
    ("O3",   "ikj",       "ikj, no -ffast-math"),
    ("fast", "ikj",       "ikj"),
    ("fast", "tile_k",    "ikj, k tiled"),
    ("fast", "tile_ijk",  "ikj, i,j,k tiled"),
]


def read_ceilings(path):
    vals = {}
    for line in open(path):
        if "=" in line and line.split("=")[0].isupper():
            key, val = line.strip().split("=", 1)
            vals[key] = float(val)
    return vals


def read_results(path):
    rows = []
    for line in open(path):
        if line.startswith("RESULT,"):
            _, build, n, kernel, ms, gf = line.strip().split(",")
            rows.append((build, int(n), kernel, float(ms), float(gf)))
    return rows


def points_for(rows, n):
    pts = []
    ai = n / 6.0  # 2n^3 FLOP over 3n^2 floats x 4 bytes
    for build, prefix, label in STEPS:
        if prefix in ("tile_k", "tile_ijk"):
            cands = [r for r in rows if r[0] == build and r[1] == n
                     and r[2].startswith(prefix) and r[2][len(prefix):].isdigit()]
            if not cands:
                continue
            best = max(cands, key=lambda r: r[4])
            pts.append((f"{label} (T={best[2][len(prefix):]})", ai, best[4]))
        else:
            for r in rows:
                if r[0] == build and r[1] == n and r[2] == prefix:
                    pts.append((label, ai, r[4]))
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("ceilings")
    ap.add_argument("--machine", default="")
    ns = ap.parse_args()

    c = read_ceilings(ns.ceilings)
    rows = read_results(ns.results)
    machine = f" — {ns.machine}" if ns.machine else ""
    for n in sorted({r[1] for r in rows}):
        roofline.plot_roofline(
            c["PEAK_BANDWIDTH"], c["PEAK_GFLOPS_SIMD_FMA"], c.get("PEAK_GFLOPS_SIMD_NOFMA"),
            c.get("PEAK_GFLOPS_SCALAR_FMA"), c.get("PEAK_GFLOPS_SCALAR_NOFMA"),
            points=points_for(rows, n),
            title=f"Matrix multiply, n = {n}{machine} (single core, FP32)",
            legend_loc="lower center",  # the kernels' labels stack on the right
            outfile=f"mm_roofline_{n}.png")


if __name__ == "__main__":
    main()
