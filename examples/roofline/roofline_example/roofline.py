#!/usr/bin/env python3
"""
Roofline model plotter.
Usage: python3 roofline.py <peak_bw_GBs> <simd_fma> [simd_nofma] [scalar_fma] [scalar_nofma]
                           [--point LABEL AI GFLOPS] ...

Example:
    python3 roofline.py 113.1 73.0 47.5 18.2 9.1 --point "std::sort 1M" 5.0 1.3
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# (color, linestyle, linewidth)
_STYLES = {
    "simd_fma":    ("#1f77b4", "-",  2.5),   # blue  solid
    "simd_nofma":  ("#9467bd", "--", 1.8),   # purple dashed
    "scalar_fma":  ("#ff7f0e", "--", 1.8),   # orange dashed
    "scalar_nofma":("#2ca02c", ":",  1.8),   # green  dotted
}


def _draw_ceiling(ax, ai, bw, peak, key, label):
    color, ls, lw = _STYLES[key]
    perf  = np.minimum(bw * ai, peak)
    ridge = peak / bw
    ax.loglog(ai, perf, color=color, lw=lw, ls=ls, zorder=3, label=label)
    ax.scatter([ridge], [peak], color=color, zorder=5, s=50)
    return ridge, color


_POINT_COLOR = "crimson"

# Fallback offsets for isolated points (not part of any AI-cluster): a fan of
# directions/magnitudes so nearby-but-unclustered points don't collide.
_LABEL_OFFSETS = [
    (1.35, 1.6), (2.3, 0.95), (1.15, 0.5),
    (3.2, 1.9), (1.15, 2.7), (4.2, 0.65),
]


# White backing behind every label: guarantees readability regardless of
# what's plotted behind it (a ceiling line, another leader line, ...)
# instead of trying to compute a y-position that dodges every line — that
# approach turned out fragile (a nudge away from one ceiling line could land
# right on a second one when two ceilings sit close together).
_LABEL_BBOX = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2)


def _place_point_labels(ax, points, color=_POINT_COLOR):
    """Draw each point's star marker plus a labeled leader line.

    Points whose AI values are close (e.g. several matmul variants at the
    same n) get their labels stacked vertically off to the side, in the same
    top-to-bottom order as their markers' GFLOPS ranking — so leader lines
    fan out monotonically and never cross. Isolated points fall back to a
    fixed offset from a rotating fan of directions.

    Returns the list of all label y-positions used, so the caller can extend
    the axes' y-limits to fit them — otherwise a label stacked above the
    highest ceiling can land outside the auto-scaled range and get clipped
    right onto that ceiling line, which looks identical to the very overlap
    this function exists to avoid.
    """
    if not points:
        return []

    label_ys = []

    # Group points into clusters by AI proximity (ratio-based, since the
    # axis is log-scaled): each cluster is a list of indices into `points`.
    # Threshold is generous (2.5x) because collisions happen between the
    # *labels* of nearby points, not just points with near-identical AI —
    # two points 1.5x apart in AI can still land close enough in pixel space
    # for their labels to overlap once both are placed.
    order = sorted(range(len(points)), key=lambda idx: points[idx][1])
    clusters = []
    for idx in order:
        ai = points[idx][1]
        for c in clusters:
            cai = points[c[0]][1]
            if cai > 0 and ai > 0 and max(ai, cai) / min(ai, cai) < 2.5:
                c.append(idx)
                break
        else:
            clusters.append([idx])

    for c in clusters:
        for idx in c:
            _, ai_pt, gf_pt = points[idx]
            ax.scatter([ai_pt], [gf_pt], marker="*", s=200, color=color,
                       zorder=7, clip_on=False)

        if len(c) == 1:
            idx = c[0]
            label, ai_pt, gf_pt = points[idx]
            dx, dy = _LABEL_OFFSETS[idx % len(_LABEL_OFFSETS)]
            label_y = gf_pt * dy
            label_ys.append(label_y)
            ax.annotate(label, xy=(ai_pt, gf_pt), xytext=(ai_pt * dx, label_y),
                        fontsize=8, color=color, bbox=_LABEL_BBOX,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
            continue

        # Cluster: stack labels vertically to the right, ordered by GFLOPS
        # so the label order matches the marker order top-to-bottom and no
        # two leader lines cross.
        c_sorted = sorted(c, key=lambda idx: points[idx][2])
        ais = [points[idx][1] for idx in c_sorted]
        gfs = [points[idx][2] for idx in c_sorted]
        label_x = max(ais) * 2.6
        log_lo, log_hi = np.log10(min(gfs) * 0.6), np.log10(max(gfs) * 1.8)
        n = len(c_sorted)
        for k, idx in enumerate(c_sorted):
            label, ai_pt, gf_pt = points[idx]
            frac = k / (n - 1) if n > 1 else 0.5
            label_y = 10 ** (log_lo + frac * (log_hi - log_lo))
            label_ys.append(label_y)
            ax.annotate(label, xy=(ai_pt, gf_pt), xytext=(label_x, label_y),
                        fontsize=8, color=color, bbox=_LABEL_BBOX,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    return label_ys


def plot_roofline(peak_bw: float,
                  simd_fma:    float,
                  simd_nofma:  float | None = None,
                  scalar_fma:  float | None = None,
                  scalar_nofma:float | None = None,
                  points: list | None = None,
                  title: str = "",
                  machine: str = "",
                  show: bool = False,
                  outfile: str = "roofline.png") -> None:
    """points: list of (label, ai, gflops) tuples to overlay on the plot."""

    ai = np.logspace(-2, 4, 2000)
    fig, ax = plt.subplots(figsize=(11, 6))

    # Draw bottom-to-top so higher lines sit on top visually
    ceilings = [
        ("scalar_nofma", scalar_nofma, f"Scalar mul-only  ({scalar_nofma:.1f} GFLOP/s)" if scalar_nofma else None),
        ("scalar_fma",   scalar_fma,   f"Scalar FMA       ({scalar_fma:.1f} GFLOP/s)"   if scalar_fma   else None),
        ("simd_nofma",   simd_nofma,   f"SIMD mul-only    ({simd_nofma:.1f} GFLOP/s)"   if simd_nofma   else None),
        ("simd_fma",     simd_fma,     f"SIMD FMA         ({simd_fma:.1f} GFLOP/s)"),
    ]

    for key, peak, label in ceilings:
        if peak is None:
            continue
        _draw_ceiling(ax, ai, peak_bw, peak, key, label)

    # Memory bandwidth label on the slope
    slope_x = 10 ** (np.log10(ai[0]) / 2 + np.log10(simd_fma / peak_bw) / 2)
    ax.text(slope_x * 0.7, peak_bw * slope_x * 4,
            f"Peak BW\n{peak_bw:.1f} GB/s",
            fontsize=9, ha="right", va="bottom", color="#555")

    # --- kernel data points ---
    label_ys = _place_point_labels(ax, points)

    ax.set_xlabel("Arithmetic Intensity  (FLOP / byte)", fontsize=12)
    ax.set_ylabel("Performance  (GFLOP/s)", fontsize=12)
    ax.set_title(title or f"Roofline Model{' — ' + machine if machine else ''} (single thread, FP32)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

    # Labels stacked above/below everything else (e.g. a cluster's top entry
    # sitting above the highest ceiling) can land outside the auto-scaled
    # y-range and get clipped right onto whatever line is at the edge —
    # extend the range to fit them, with a little breathing room.
    y_values = [v for v in (simd_fma, simd_nofma, scalar_fma, scalar_nofma, peak_bw * ai[0]) if v] + list(label_ys)
    if points:
        y_values += [gf for _, _, gf in points]
    if y_values:
        y_lo, y_hi = min(y_values), max(y_values)
        ax.set_ylim(y_lo / 1.6, y_hi * 1.6)
    ax.set_xlim(ai[0], ai[-1])

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("peak_bw",      type=float)
    parser.add_argument("simd_fma",     type=float)
    parser.add_argument("simd_nofma",   type=float, nargs="?")
    parser.add_argument("scalar_fma",   type=float, nargs="?")
    parser.add_argument("scalar_nofma", type=float, nargs="?")
    parser.add_argument("--point", nargs=3, action="append",
                        metavar=("LABEL", "AI", "GFLOPS"),
                        help="overlay a kernel point; may be repeated")
    parser.add_argument("--title", default="")
    parser.add_argument("--machine", default="",
                        help="CPU name for the default title (the Makefile "
                             "passes sysctl's machdep.cpu.brand_string)")
    parser.add_argument("--show", action="store_true",
                        help="open an interactive window (blocks until closed); "
                             "off by default so this can run headless")
    ns = parser.parse_args()

    pts = [(lbl, float(ai), float(gf)) for lbl, ai, gf in (ns.point or [])]
    plot_roofline(ns.peak_bw, ns.simd_fma, ns.simd_nofma,
                  ns.scalar_fma, ns.scalar_nofma,
                  points=pts, title=ns.title, machine=ns.machine, show=ns.show)
