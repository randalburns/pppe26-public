#!/usr/bin/env python3
"""
Generate the Activity 5 data set: a tree of files with duplicates in it.

Usage: python3 gen_data.py [OUT] [--dirs 8] [--mb 768] [--dup 0.25] [--seed 5]

OUT/d0 ... OUT/d7 are the top-level directories, one per producer in Part 2.
Each holds a few levels of subdirectories. File sizes are spread from 4 KB to
8 MB (log-uniform, so most files are small and most bytes are in large ones).
About a quarter of the files are exact copies of another file, placed in a
random directory, often a different one from the original. The same seed
always produces the same tree.
"""

import argparse
import math
import os
import random
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default="data")
    ap.add_argument("--dirs", type=int, default=8, help="top-level directories")
    ap.add_argument("--mb", type=int, default=768, help="approximate total size")
    ap.add_argument("--dup", type=float, default=0.25, help="fraction of files that are copies")
    ap.add_argument("--seed", type=int, default=5)
    ns = ap.parse_args()

    rng = random.Random(ns.seed)
    if ns.dirs < 1 or ns.mb < 1 or not 0 <= ns.dup <= 1:
        ap.error("use positive --dirs and --mb and 0 <= --dup <= 1")
    try:
        os.makedirs(ns.out, exist_ok=False)
    except FileExistsError:
        ap.error(f"{ns.out} already exists; choose a new output directory")

    # A few nested subdirectories under each top-level directory.
    dirs = []
    for d in range(ns.dirs):
        top = os.path.join(ns.out, f"d{d}")
        dirs.append(top)
        for a in range(3):
            sub = os.path.join(top, f"s{a}")
            dirs.append(sub)
            for b in range(2):
                dirs.append(os.path.join(sub, f"t{b}"))
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    budget = ns.mb * 1024 * 1024
    written, nfiles, ncopies = 0, 0, 0
    originals = []
    while written < budget:
        where = rng.choice(dirs)
        path = os.path.join(where, f"f{nfiles:05d}.bin")
        if originals and rng.random() < ns.dup:
            src = rng.choice(originals)
            shutil.copyfile(src, path)
            size = os.path.getsize(src)
            ncopies += 1
        else:
            size = int(math.exp(rng.uniform(math.log(4 * 1024), math.log(8 * 1024 * 1024))))
            with open(path, "wb") as f:
                f.write(rng.randbytes(size))
            originals.append(path)
        written += size
        nfiles += 1

    print(f"{ns.out}: {nfiles} files in {len(dirs)} directories, "
          f"{written / 1048576:.0f} MB, {ncopies} copies")


if __name__ == "__main__":
    main()
