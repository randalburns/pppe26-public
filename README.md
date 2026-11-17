# Parallel Programming and Performance Engineering

Course materials for **PPPE** — lectures, runnable examples, and activities for a
hands-on course on making code use the machine well.

## Repository layout

```
course_materials/    lecture notes and notebooks, numbered in order
examples/            one directory per topic, each with a README and runnable code
```

### Lectures

[Class Sessions in Presentation Order](course_materials/)

Each number session does not necessarily corerspond to a single day of class. We will consume material at the pace appropriate for class.

| | |
|---|---|
| [00.Intro.ipynb](course_materials/00.Intro.ipynb) | why performance is hard; modern processors |
| [01.pipeline.compiler_optimization.md](course_materials/01.pipeline.compiler_optimization.md) | the processor pipeline, RAW hazards, and what `-O0`–`-O3` actually do |
| [02.speedup.ipynb](course_materials/02.speedup.ipynb) | speedup and Amdahl's law |
| [03.ILP.md](course_materials/03.ILP.md) | instruction-level parallelism, CPI/IPC, the reorder buffer, branch prediction |
| [04.branch_optimization.md](course_materials/04.branch_optimization.md) | branchless code, lookup tables, and the two shapes of "the compiler will not do this" |
| [05.memoryhierarchy.md](course_materials/05.memoryhierarchy.md) | caches, lines, sets and the cost of a miss |
| [06.loops.md](course_materials/06.loops.md) | the five loop transformations, and naming the bottleneck before picking one |
| [07.vectorization.md](course_materials/07.vectorization.md) | vector registers, lanes, and the three ways to program a SIMD unit |
| [09.threads.md](course_materials/09.threads.md) | processes and threads: the OS abstractions behind parallelism, and what they share |

### Examples

Each directory stands alone: a `README.<topic>.md` explaining the idea, source you can compile,
and (where it matters) measured numbers.

**Single core**

| | |
|---|---|
| [pipeline/](examples/pipeline/) | pipeline stalls, dependency chains, CSE and dead-code elimination |
| [ILP/](examples/ILP/) | out-of-order and speculative execution, separating dependent instructions |
| [loop_optimizations/](examples/loop_optimizations/) | unrolling, fusion, fission, interchange, tiling |
| [branch_optimizations/](examples/branch_optimizations/) | branch-free code, lookup tables, aliasing-blocked unswitching |
| [strength.reduction/](examples/strength.reduction/) | replacing expensive operations with cheap ones |
| [vectorization/](examples/vectorization/) | SIMD by hand and via Highway (AVX2 / NEON) |
| [sorting/](examples/sorting/) | why `std::sort` is built the way it is — the culminating single-core example |

**Memory**

| | |
|---|---|
| [memory_hierarchy/](examples/memory_hierarchy/) | measured cache latency, bandwidth, set conflicts, core-to-core transfer |
| [virtual_memory/](examples/virtual_memory/) | page tables, TLB reach, and the TLB cliff |
| [sharing/](examples/sharing/README.sharing.md) | what sharing a cache line costs: true sharing, false sharing, and where the threads run |
| [prefetch.example/](examples/prefetch.example/) | why software prefetching mostly doesn't help |
| [roofline/](examples/roofline/) | arithmetic intensity as the guide to what is worth parallelizing |
| [matmul/](examples/matmul/README.matmul.md) | matrix multiply step by step: flags, loop order, tiling, on the roofline |

**Parallelism**

| | |
|---|---|
| [openmp/](examples/openmp/) | parallel regions, parallel loops, reductions, tasks |
| [cilk/](examples/cilk/) | OpenCilk primitives, work stealing, work/span |
| [prefix_sum/](examples/prefix_sum/README.prefix_sum.md) | a loop no pragma can parallelize: a parallel scan that does more work, and the memory bandwidth that limits it |
| [java_threads/](examples/java_threads/) | fork/join, and the startup cost of a thread pool |
| [synchronization/](examples/synchronization/README.synchronization.md) | one shared counter, incremented eight ways: races, locks, atomics, and not sharing |
| [ordering/](examples/ordering/README.ordering.md) | making one thread wait for another: a bounded blocking queue six ways, barriers, futures, and deadlock |
| [joblib/](examples/joblib/) | parallelism from Python, and the hazard of parallelizing bad code |
| [java_rag_pipeline/](examples/java_rag_pipeline/) | an end-to-end pipeline to profile and speed up |


## Activities

- [Activity 4 — Load Balancing Adaptive Simpson Integration](activities/activity4/)

## Getting started

Python environment for the notebooks:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

The C and C++ examples build with the system compiler and no configuration:

```bash
cd examples/memory_hierarchy && make
cd examples/pipeline && ./buildandrun.sh     # builds and runs at -O0 through -O3
```

Toolchains, only needed for the directories that use them:

* **OpenMP** — `brew install libomp` on macOS; already present with `gcc` on Linux.
* **OpenCilk** — install from [opencilk.org](https://www.opencilk.org). The
  [cilk/](examples/cilk/) Makefile expects it at `~/opencilk`.
* **Java** — JDK 17+ and Maven for [java_rag_pipeline/](examples/java_rag_pipeline/).

These examples have been tested on at least two machines (Apple M5, MacOSX using clang 17.0, and AMD Ryzen, Pop!_OS 24.04 LTS using clang 18.1 and g++ 13.3). They will not necessarily run on your machine. You can ask Claude or another tool to adapt them to your machine and that should work.  In past versions of the course, I have distributed Docker files that provide a uniform execution environment. I think that is now more difficult and less reliable.

This means that the results you get from examples and activities will vary with software, hardware, etc. They many even vary widely, i.e. not have the same form. This is interesting and something to raise with the instructor and TAs. 

## A note on the numbers

Performance claims in this repository are measured, not estimated, and measurements are
machine-specific. Results were taken on an Apple Silicon Mac and on an AMD Ryzen AI 9
HX 370; where the two disagree in an interesting way, both are reported (files ending
in `.ryzen.md` hold the second machine's run). **Expect your own numbers to differ.**
Reproducing a measurement on your hardware and explaining why it came out differently
is a large part of the point.

## AI policy

The course is designed around working with an AI partner, and much of the prose in
these materials was drafted with Claude and edited by Randal Burns; individual files
say so where it applies. Use any tool you want unless a specific assignment says
otherwise. Exams and quizzes ask for written answers and pseudo-code in your own words.
See the syllabus for the full policy.
