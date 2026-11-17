# Roofline Model

The [roofline model](https://en.wikipedia.org/wiki/Roofline_model) (Williams,
Waterman, Patterson, 2009) bounds a kernel's achievable performance by the
smaller of two ceilings: how fast the processor can compute, and how fast
memory can feed it data. Which ceiling applies depends on **arithmetic
intensity** — FLOPs per byte moved from DRAM. Low-intensity kernels (a dot
product, a sparse matvec) are memory-bound no matter how fast the ALUs are;
high-intensity kernels (a blocked matmul) are compute-bound no matter how
much bandwidth is available.

The lecture is [16.roofline.md](../../course_materials/16.roofline.md),
which presents this directory's results.

## 1 · The concept, first

The model itself — operational intensity, the ridge point, what multicore
does to it, and where GPUs and TPUs sit — is Part 1 of
[16.roofline.md](../../course_materials/16.roofline.md). Its numbers
(16 GB/s, 16 GFLOP/s) are illustrative, not measured — that's what
`roofline_example/` is for.

[roofline_lecture/roofline_overview.ipynb](roofline_lecture/roofline_overview.ipynb) is the original notebook
Part 1 was lifted from, kept for reference. Its plotting cells are R
(`ggplot2`) with the output saved, so it reads with no setup; to run them,
install R and the `ir` kernel:

```bash
brew install r
Rscript -e 'install.packages(c("ggplot2", "repr", "IRkernel"), repos="https://cloud.r-project.org")'
Rscript -e 'IRkernel::installspec(name = "ir", displayname = "R")'
```

## 2 · `roofline_example/` — the same model, measured on real hardware

A hand-written, single-threaded FP32 benchmark suite that builds an actual
roofline plot for the machine it runs on (Apple M5), instead of the
lecture's toy numbers.

```bash
cd roofline_example
make plot
```

builds everything, runs each benchmark, and writes `roofline.png`: four
compute ceilings (scalar/SIMD × with/without FMA, from `peak_flops.cpp`), a
memory-bandwidth slope (from `stream.cpp`), and a star for each kernel below.

| Kernel | Source | What it shows |
|---|---|---|
| dot product | `dot_bench.cpp` | memory-bound (AI = 0.25); also where a real bug lived — see below |
| SAXPY | `saxpy_bench.cpp` | classic BLAS-1 memory-bound point, next to dot product |
| 7-point stencil | `stencil_bench.cpp` | memory-bound in theory, lands *above* its own naive bandwidth line — real cache-line reuse the byte-counting model doesn't capture |
| SpMV | `spmv_bench.cpp` | low AI *and* underperforms even that — irregular gather access, not just bandwidth, is the bottleneck |
| `std::sort` | `sort_bench.cpp` | comparison-bound, far below every ceiling |
| N-body | `nbody_bench.cpp` | the opposite end of the spectrum — compute-bound, unbounded AI as N grows |
| naive matmul ×4 | `matmul_naive_bench.cpp` | scalar / scalar+FMA / SIMD / SIMD+FMA, hand-blocked to actually reach the ceilings — see below |
| sgemm (Accelerate/OpenBLAS) | `matmul_bench.cpp` | built and runnable standalone, deliberately **not** plotted |

Three things worth knowing before reading the source, all found by measuring
rather than assuming, and documented in the relevant file's header comment:

- **`dot_bench.cpp`** originally reported `inf` GFLOP/s. The compiler proved
  the timed function was pure and CSE'd away 4 of its 5 timed calls. Fixed
  with an `asm volatile` barrier — the same trick used throughout this suite
  to keep the compiler from fusing or vectorizing things it shouldn't.
- **`matmul_bench.cpp`'s sgemm is excluded from the plot on purpose.** On
  Apple Silicon it runs on the AMX matrix coprocessor, not the NEON ALU
  pipeline `peak_flops.cpp` measures — comparing it against these ceilings
  would be comparing against the wrong hardware resource.
- **`matmul_naive_bench.cpp`'s four variants needed real register blocking**,
  not just vectorization, to reach their ceilings. An unblocked version was
  L1-bandwidth-bound; a 1-column SIMD tile made FMA measure *slower* than
  plain multiply-add, which turned out to be a tile-shape problem (no B
  reuse across columns), not a scheduling or latency issue — confirmed by
  testing a hand-scheduled assembly version that changed nothing. Widening
  to a proper 4×4 tile got SIMD+FMA to about 90% of its ceiling.

`roofline.py` does the plotting — log-log ceilings plus each kernel as a
labeled point, with cluster-aware label placement so points that land close
together (several matmul variants at the same n, or several memory-bound
kernels near the ridge) stack cleanly instead of overlapping.

## 3 · `roofline_example/` again — the same kernels, all cores

The lecture's own multicore section makes a specific, testable claim: adding
cores raises the compute ceiling but not the (shared) memory-bandwidth
ceiling, so the ridge point moves right and low-AI kernels stop benefiting
from more cores. Every kernel above has a `_mt` companion that partitions
its work across `std::thread::hardware_concurrency()` threads (10 on this
M5: 4 P-cores + 6 E-cores) to test that claim directly, plus `stream_mt.cpp`
and `peak_flops_mt.cpp` to measure the actual multicore ceilings rather than
assuming they scale linearly.

```bash
cd roofline_example
make plot_mt
```

writes `roofline_multicore.png`, plotted by `roofline_multicore.py` (a thin
wrapper that reuses `roofline.py`'s ceiling-drawing and label-placement code
rather than duplicating it — same plot logic, different title/output file).

Measured on this machine, single-thread vs. 10-thread:

| Kernel | AI | 1 thread | 10 threads | Scaling |
|---|---:|---:|---:|---:|
| Peak bandwidth (`stream_mt`) | — | 116.7 GB/s | 123.6 GB/s | **1.06x** |
| SIMD FMA ceiling (`peak_flops_mt`) | — | 134.6 | 857.4 GFLOP/s | **6.37x** |
| saxpy 32M | 0.167 | 20.2 | 22.6 | 1.12x |
| `std::sort` 16M | 6.66 | 1.5 | 2.1 | 1.35x |
| dot product 32M | 0.25 | 20.8 | 32.4 | 1.56x |
| 7-pt stencil 256³ | 0.25 | 37.2 | 81.2 | 2.18x |
| SpMV 4M×7 | 0.167 | 2.3 | 5.8 | 2.56x |
| naive mm (4 variants) | 85.3 | 17–121 | 75–441 | 3.6–4.7x |
| N-body 4096 | 5632 | 89.1 | 287.9 | 3.23x |

N-body is the noisy one: each timed run lasts about 4 ms, and three runs of
the suite have measured 58, 65 and 89 GFLOP/s single-threaded. Run on a quiet
machine (`top -o cpu` first): a process holding the P-cores pushes everything
onto E-cores.

Confirms the lecture's claim, with the nuance a naive "memory-bound = no
multicore benefit" summary would miss:

- **Bandwidth is already saturated by roughly one core on this chip.**
  `stream_mt`'s aggregate bandwidth (124 GB/s) is barely above `stream`'s
  single-core number (117 GB/s) — this Apple Silicon core can already drive
  close to the full memory controller alone, so saxpy and dot product, which
  are purely sequential streams, gain almost nothing from more cores.
- **SpMV and the stencil do better than pure streaming, for different
  reasons.** SpMV's irregular `x[col_idx[k]]` gathers are limited by a
  single core's memory-level parallelism (how many outstanding loads it can
  keep in flight), not raw bandwidth — more cores issuing independent
  gathers recovers some of that gap (2.56x) even though the kernel is still
  memory-bound. The stencil's extra scaling (2.18x) comes from the same
  cache-line reuse that already put it above its naive bandwidth line
  single-threaded.
- **Compute-bound kernels scale 3.2–4.7x, not 10x**, because 6 of the 10
  cores are E-cores with a fraction of a P-core's throughput each — the
  aggregate ceiling is real, but it isn't `10 x single-core` (the peak
  itself is 6.37x). The kernels fall short even of that: N-body reaches
  3.23x and SIMD+FMA matmul 3.64x, 51% of its multicore ceiling against
  90% on one core.
- **`sort_bench_mt.cpp` needed a structurally different algorithm**, not
  just a partitioned loop: `std::sort` has no thread-count argument, so it
  chunk-sorts per thread and merges the sorted chunks with a sequential
  pairwise `std::inplace_merge` reduction. That serial merge tail is why
  sort only reaches 1.35x — a real example of Amdahl's law, not a tuning
  gap. (It also needed its own comparison-counting instrumentation run,
  since a different algorithm does a different number of comparisons than
  the single-thread version — its AI/GFLOPS figures aren't just the
  single-thread ones divided by time.)

All `_mt` kernels reuse their single-thread counterpart's already-verified
inner loop bodies (copied, not re-derived) — only the threading/partitioning
wrapper (`mt_common.hpp`) is new, so none of the correctness or codegen
verification from the single-thread files needed repeating.

## The flow

Read the lecture for the *shape* of the argument — why intensity determines
the bound, why that bound moved as hardware changed, and what multicore does
to the ridge point. Then run the single-thread example and watch that shape
show up as measured GFLOP/s and GB/s on the machine in front of you,
including the places real code doesn't behave like the idealized model —
cache reuse the byte-counting AI formula misses, access patterns that miss
bandwidth by more than intensity predicts, and a compute unit (AMX) the
model was never built to describe at all. Then run the multicore version and
watch the lecture's multicore claim — compute ceiling scales, bandwidth
ceiling doesn't, so the ridge point moves right — play out as a measured
6.4x compute ceiling against a 1.06x bandwidth one, on real kernels instead
of a toy example.
