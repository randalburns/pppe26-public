// Build: clang++ -std=c++17 -O3 -march=native -o sort_bench_mt sort_bench_mt.cpp
// Multicore companion to sort_bench.cpp. std::sort itself has no thread-count
// argument — there's no loop to just partition the way the other _mt files
// do. The standard approach: split into N chunks, sort each chunk on its own
// thread (embarrassingly parallel), then merge the sorted chunks together.
//
// The merge here is a straightforward sequential N-way reduction (pairwise
// std::inplace_merge, halving the chunk count each round) — it is NOT
// parallelized. For N=16M elements and 10 threads, the parallel sort phase
// dominates, but the serial merge tail means this measures something
// structurally different from "linear speedup," and that's the honest
// result to report, not something to engineer away for a better number.

#include "mt_common.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <functional>
#include <random>
#include <thread>
#include <vector>

static const int NTIMES = 5;

// Chunk-sort on `nthreads` threads, then a sequential pairwise-merge reduction
// (halving the run count each round) down to one fully sorted range. `comp`
// is threaded through both phases so the same function can be used for the
// timed run (default `<`) and a comparison-counting instrumented run.
template <typename Compare>
static void parallel_sort(std::vector<float>& data, const std::vector<size_t>& bounds,
                           unsigned nthreads, Compare comp) {
    run_on_threads(nthreads, [&](unsigned tid, unsigned) {
        std::sort(data.begin() + bounds[tid], data.begin() + bounds[tid + 1], comp);
    });
    std::vector<size_t> runs(bounds);
    while (runs.size() > 2) {
        std::vector<size_t> next;
        next.push_back(runs[0]);
        for (size_t i = 0; i + 2 < runs.size(); i += 2) {
            std::inplace_merge(data.begin() + runs[i], data.begin() + runs[i + 1],
                                data.begin() + runs[i + 2], comp);
            next.push_back(runs[i + 2]);
        }
        if ((runs.size() - 1) % 2 == 1) next.push_back(runs.back());
        runs = next;
    }
}

int main() {
    unsigned nthreads = std::thread::hardware_concurrency();
    size_t N = 16'000'000; // same as sort_bench.cpp
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(N);

    std::vector<size_t> bounds(nthreads + 1);
    for (unsigned t = 0; t <= nthreads; t++) {
        size_t b, e;
        thread_range(N, t < nthreads ? t : nthreads - 1, nthreads, b, e);
        bounds[t] = (t < nthreads) ? b : N;
    }

    // --- Timing: best of NTIMES runs, fresh random data each time ---
    double best_t = 1e9;
    for (int k = 0; k < NTIMES; k++) {
        for (auto& x : data) x = dist(rng);
        auto t0 = std::chrono::high_resolution_clock::now();
        parallel_sort(data, bounds, nthreads, std::less<float>());
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        if (dt < best_t) best_t = dt;
    }
    bool sorted = std::is_sorted(data.begin(), data.end());

    // --- Count comparisons across both phases (instrumented run, separate
    // from timing) so this has an AI/GFLOPS figure in the same units as
    // sort_bench.cpp — this is a structurally different algorithm (chunk
    // sort + merge), so its comparison count genuinely differs from the
    // single-thread version's and has to be measured fresh, not reused.
    std::atomic<long long> cmp_count{0};
    for (auto& x : data) x = dist(rng);
    parallel_sort(data, bounds, nthreads, [&cmp_count](float a, float b) {
        cmp_count.fetch_add(1, std::memory_order_relaxed);
        return a < b;
    });

    double bytes  = (double)N * sizeof(float);
    double ai     = (double)cmp_count / bytes;
    double gflops = (double)cmp_count / best_t / 1e9;

    printf("std::sort (MT)  N=%zu floats, %u threads, chunk-sort + serial merge (best of %d)\n",
           N, nthreads, NTIMES);
    printf("  Time:        %.3f ms\n",        best_t * 1000.0);
    printf("  Sorted:      %s\n",             sorted ? "yes" : "NO -- BUG");
    printf("  Comparisons: %lld\n",           (long long)cmp_count);
    printf("  Arith. Int.: %.4f FLOP/byte\n", ai);
    printf("  Performance: %.4f GFLOP/s\n",  gflops);
    printf("SORT_AI_MT=%.4f\n",     ai);
    printf("SORT_GFLOPS_MT=%.4f\n", gflops);
    printf("MT_THREADS=%u\n", nthreads);
    return 0;
}
