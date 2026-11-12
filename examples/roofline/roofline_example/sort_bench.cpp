// Build: clang++ -std=c++17 -O3 -o sort_bench sort_bench.cpp
// Measures arithmetic intensity and performance of std::sort on 16M floats.
//
// Arithmetic intensity = comparisons / (N × sizeof(float))
//   - comparisons counted via a custom lambda (one instrumented run)
//   - bytes = dataset size (N × 4): the data that must be touched to sort it
//   - this is an approximation, not true DRAM traffic: introsort's recursion
//     only re-reads from DRAM at the top few levels, before partitions shrink
//     below cache size; once a partition is cache-resident, further passes
//     over it are cache hits, not DRAM accesses. Treat the plotted AI as a
//     rough lower bound, not an exact byte count.
//
// N defaults to 64 MB (> the M5's 12 MB SLC) so the initial pass is a real
// DRAM read rather than entirely cache-resident, as with the other kernels.
//
// Performance = comparisons / best_time, expressed in GFLOP/s
//   - timed separately (no counter overhead) over multiple runs

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace std::chrono;

static const int NTIMES = 10;

int main(int argc, char* argv[]) {
    size_t N = argc > 1 ? (size_t)std::atoll(argv[1]) : 16'000'000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(N);

    // --- Timing: best of NTIMES runs, fresh random data each time ---
    double best_t = 1e9;
    for (int k = 0; k < NTIMES; k++) {
        for (auto& x : data) x = dist(rng);
        auto t0 = high_resolution_clock::now();
        std::sort(data.begin(), data.end());
        auto t1 = high_resolution_clock::now();
        double t = duration<double>(t1 - t0).count();
        if (t < best_t) best_t = t;
    }

    // --- Count comparisons (instrumented run, separate from timing) ---
    long long cmp_count = 0;
    for (auto& x : data) x = dist(rng);
    std::sort(data.begin(), data.end(), [&cmp_count](float a, float b) {
        ++cmp_count;
        return a < b;
    });

    double bytes  = (double)N * sizeof(float);
    double ai     = (double)cmp_count / bytes;          // comparisons / byte
    double gflops = (double)cmp_count / best_t / 1e9;  // Gcmp/s

    printf("std::sort  N=%zu floats (best of %d runs)\n", N, NTIMES);
    printf("  Time:        %.3f ms\n",   best_t * 1000.0);
    printf("  Comparisons: %lld\n",      cmp_count);
    printf("  Arith. Int.: %.4f FLOP/byte\n", ai);
    printf("  Performance: %.4f GFLOP/s\n",  gflops);
    printf("SORT_AI=%.4f\n",     ai);
    printf("SORT_GFLOPS=%.4f\n", gflops);
    return 0;
}
