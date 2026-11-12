#!/usr/bin/env python3
"""
Multicore roofline model plotter — thin wrapper around roofline.py's
plot_roofline(), same ceiling/point/label-placement logic, just a different
default title and output filename so the single-thread and multicore plots
don't clobber each other.

Usage: python3 roofline_multicore.py <peak_bw_GBs> <simd_fma> [simd_nofma] [scalar_fma] [scalar_nofma]
                                      --threads N [--point LABEL AI GFLOPS] ...

Example:
    python3 roofline_multicore.py 117.5 841.5 422.2 213.3 107.8 --threads 10 \\
        --point "dot product 32M" 0.25 32.7
"""

import argparse
import roofline


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("peak_bw",      type=float)
    parser.add_argument("simd_fma",     type=float)
    parser.add_argument("simd_nofma",   type=float, nargs="?")
    parser.add_argument("scalar_fma",   type=float, nargs="?")
    parser.add_argument("scalar_nofma", type=float, nargs="?")
    parser.add_argument("--threads", type=int, required=True,
                         help="core count these aggregate numbers were measured on")
    parser.add_argument("--point", nargs=3, action="append",
                         metavar=("LABEL", "AI", "GFLOPS"),
                         help="overlay a kernel point; may be repeated")
    parser.add_argument("--title", default="")
    parser.add_argument("--machine", default="",
                         help="CPU name for the default title")
    parser.add_argument("--cores", default="",
                         help="core mix for the default title, e.g. 4P+6E")
    parser.add_argument("--show", action="store_true",
                         help="open an interactive window (blocks until closed); "
                              "off by default so this can run headless")
    ns = parser.parse_args()

    pts = [(lbl, float(ai), float(gf)) for lbl, ai, gf in (ns.point or [])]
    machine = f" — {ns.machine}" if ns.machine else ""
    cores   = f", {ns.cores}" if ns.cores else ""
    title = ns.title or f"Roofline Model{machine} ({ns.threads} threads{cores}, FP32, aggregate)"
    roofline.plot_roofline(ns.peak_bw, ns.simd_fma, ns.simd_nofma,
                            ns.scalar_fma, ns.scalar_nofma,
                            points=pts, title=title, show=ns.show,
                            outfile="roofline_multicore.png")


if __name__ == "__main__":
    main()
