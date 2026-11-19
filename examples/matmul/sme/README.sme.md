# Add-on: Matrix Multiply on the Matrix Unit (Arm SME)

Lecture 17 ends with register tiling at 114 GFLOP/s, 87% of what one M5 core
can do with ordinary NEON vector instructions. This add-on writes the same
multiply for the M5's **matrix unit**, using Arm's SME instructions, and
reaches 1,870 GFLOP/s on one thread: level with Apple's own library.

For what a matrix unit is, and how Apple's compares with Intel's and
NVIDIA's, read [../matrix_units.html](../matrix_units.html) first.

**Apple silicon M4 or later only.** Other machines have no SME; `make` says
so and stops.

## Source

  * [mm_sme.cpp](mm_sme.cpp) — three kernels, a peak-rate test, and a
    comparison with Accelerate's `sgemm`
  * [packing.html](packing.html) — a self-contained page on why and how the
    kernel packs A and B, with figures. Open it in a browser
  * [Makefile](Makefile) — `make run` writes
    [results_sme.M5.csv](results_sme.M5.csv), in about 15 s

```bash
make run
./mm_sme peak                           # the unit's peak rate
./mm_sme 1024 sme1 sme4 sme4_pack_k1024 accelerate
```

The code uses the C intrinsics in `arm_sme.h`. It needs `-march=native+sme`:
Apple clang does not turn SME on with `-march=native` alone. (An editor that
doesn't know the flag will underline the SME code as errors.)

## SME in a paragraph

One instruction, `FMOPA`, takes two vectors of 16 floats, `a` and `b`, and
adds `a[r] * b[c]` into element (r, c) of a 16 × 16 **tile**: 256
multiply-adds. The tiles are storage inside the matrix unit, and with 32-bit
floats there are four. For C = A × B, one step of k adds (column k of A) ×
(row k of B) to C, which is one `FMOPA` for each 16 × 16 block of C. It is
the loop from Step 5 of the lecture, with a tile in place of `acc`.

A function must be in *streaming mode* to use the unit. In the code,
`__arm_locally_streaming` switches modes on entry and exit.

## The design problem: columns of A

A row of B is 16 floats next to each other in memory. A column of A is not:
its elements are a whole row apart, and streaming mode has no instruction to
gather them. So A has to be rearranged before the unit can use it. The three
kernels differ mainly in how.

| kernel | block of C | how A and B are prepared |
|---|---|---|
| `sme1` | 16 × 16, one tile | all of A is transposed first; B is read in place |
| `sme4` | 32 × 32, four tiles | the same |
| `sme4_pack_k<K>` | 32 × 32, four tiles | A and B are packed a slab of K values of k at a time, inside the matrix unit |

## Results

GFLOP/s, one thread, Apple M5, best of 10 runs:

| | n = 1024 | n = 2048 |
|---|---:|---:|
| register tiled NEON (Lecture 17) | 114 | 101 |
| `sme1` | 482 | 158–218 |
| `sme4` | 1,386 | 412–486 |
| `sme4_pack_k256` | 1,631 | 1,412 |
| `sme4_pack_k512` | 1,768 | 1,463 |
| `sme4_pack_k1024` | **1,866** | **1,696** |
| Accelerate `sgemm`, 1 thread | 1,817 | 1,860 (1,380–1,870 across runs) |

Peak rate of the unit, with no memory access: 540, 1,080 and 2,160 GFLOP/s
with 1, 2 and 4 tiles.

## What the numbers say

**Four tiles, four times the speed.** The peak test does nothing but
`FMOPA`s on the same two vectors. It doubles from 1 tile to 2 and again from
2 to 4. An `FMOPA` into a tile has to wait for the previous one into the
same tile, so tiles are independent accumulators, like the 16 registers in
Step 5 of the lecture. `sme1` reaches 89% of the one-tile peak and cannot go
further. `sme4` also uses each loaded vector twice: 4 loads feed 4 `FMOPA`s.

**Transposing the whole matrix is too expensive.** At 2 TFLOP/s the entire
multiply at n = 1024 takes 1 ms, so "only n² work" is no longer small:

| | n = 1024 | n = 2048 |
|---|---:|---:|
| transpose all of A | 0.35 ms | 11.1 ms |
| the best whole multiply | 1.15 ms | 10.1 ms |

At n = 1024 the transpose is a quarter of `sme4`'s time. At n = 2048 the
transpose alone takes longer than the best multiply. Its reads and writes
are 8 KB apart, a power of two, so they land in the same few cache sets:
the transpose problem from Lecture 6.

**Pack small pieces, and do it in the unit.** `sme4_pack` rearranges only
what it is about to use, a slab of K values of k, into two buffers that the
kernel then reads in order. A tile can be loaded by rows and stored by
columns, so the unit itself transposes a 32 × 16 piece of A with 32 vector
loads and 32 vector stores. That takes the kernel from 1,386 to 1,866
GFLOP/s at n = 1024, and from under 500 to 1,696 at n = 2048.

A first version of the packing used ordinary scalar loops outside streaming
mode. Packing was then 30–40% of the run time and the kernel ran at
1,050–1,130 GFLOP/s. Packing with the unit's own vector loads and stores is
what made the difference.

**Bigger slabs are better, up to a point.** Every slab loads and stores
each tile of C once, and that is not cheap, so K = 1024 beats 512 and 256.
At n = 2048, K = 2048 is slower again (about 1,135 GFLOP/s): the packed
buffers are then 16 MB each and no longer fit in the L2.

**Against the library.** At n = 1024 `sme4_pack_k1024` is level with
Accelerate on one thread, and at n = 2048 it reaches about 90% of it. It is
at 86% and 79% of the unit's peak. A library would also handle sizes that
are not a multiple of 32, choose K for the machine, and use more than one
thread.

## Things to try

  * run `./mm_sme peak`. Why does the rate double with each doubling of
    tiles? What does that tell you about how long one `FMOPA` takes?
  * `sme4_pack_k64` is much slower than `sme4_pack_k1024`. Count how many
    times each tile of C is loaded and stored in each
  * run `./mm_sme 4096 sme4_pack_k1024 accelerate`. Does the gap to the
    library grow? What would you tile next?
