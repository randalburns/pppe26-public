// Build: clang++ -std=c++17 -O3 -march=native -o saxpy_bench_mt saxpy_bench_mt.cpp
// Multicore companion to saxpy_bench.cpp: same memory-bound-kernel-vs-shared-
// bandwidth question as dot_bench_mt.cpp, for a pure elementwise kernel
// (no reduction, so no partial-sum combination step at all — the cleanest
// possible "does this scale with cores" test).

#include "mt_common.hpp"
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

template <typename T>
static void sink(T const& v) { asm volatile("" : : "m"(v) : "memory"); }

static const size_t N      = 32ULL * 1024 * 1024; // 32M floats, same total as saxpy_bench.cpp
static const int    NTIMES = 5;

__attribute__((noinline))
static void saxpy(float a, const float* __restrict__ x, float* __restrict__ y, size_t n) {
    asm volatile("" : : : "memory");
    for (size_t i = 0; i < n; i++) y[i] = a * x[i] + y[i];
}

int main() {
    unsigned nthreads = std::thread::hardware_concurrency();
    std::vector<float> x(N), y(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x) v = dist(rng);
    for (auto& v : y) v = dist(rng);

    double best_t = best_of_mt(nthreads, NTIMES, [&](unsigned tid, unsigned nt) {
        size_t begin, end;
        thread_range(N, tid, nt, begin, end);
        saxpy(2.0f, x.data() + begin, y.data() + begin, end - begin);
    });
    sink(y[0]);

    double flops  = 2.0 * N;
    double bytes  = 3.0 * N * sizeof(float);
    double ai     = flops / bytes;
    double gflops = flops / best_t / 1e9;

    printf("saxpy (MT)  N=%zuM floats, %u threads (best of %d)\n", N >> 20, nthreads, NTIMES);
    printf("  Time:        %.3f ms\n",        best_t * 1000.0);
    printf("  Arith. Int.: %.4f FLOP/byte\n", ai);
    printf("  Performance: %.4f GFLOP/s\n",  gflops);
    printf("SAXPY_AI_MT=%.4f\n",     ai);
    printf("SAXPY_GFLOPS_MT=%.4f\n", gflops);
    printf("MT_THREADS=%u\n", nthreads);
    return 0;
}
