// Build: clang++ -std=c++17 -O3 -march=native -o dot_bench_mt dot_bench_mt.cpp
// Multicore companion to dot_bench.cpp: partitions the same 32M-float dot
// product across all cores (each thread reduces its own disjoint slice with
// the same verified 4-accumulator NEON kernel, partial sums combined at the
// end), measuring whether a memory-bound kernel actually benefits from more
// cores once a single core already saturates DRAM bandwidth (stream_mt.cpp
// found it does not, for streaming access).
//
// Same working-set-per-thread convention as the single-thread file: each
// thread's slice is large enough that its reads come from DRAM, not cache.

#include "mt_common.hpp"
#include <arm_neon.h>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

template <typename T>
static void sink(T const& v) { asm volatile("" : : "m"(v) : "memory"); }

static const size_t N      = 32ULL * 1024 * 1024; // 32M floats, same total as dot_bench.cpp
static const int    NTIMES = 5;

// Identical to dot_bench.cpp's dot(): 4 independent NEON accumulators so the
// FMA chain latency doesn't bottleneck the loop ahead of memory bandwidth.
__attribute__((noinline))
static float dot(const float* __restrict__ a, const float* __restrict__ b, size_t n) {
    asm volatile("" : : : "memory");
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f),
                acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),      vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

int main() {
    unsigned nthreads = std::thread::hardware_concurrency();
    std::vector<float> a(N), b(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : a) x = dist(rng);
    for (auto& x : b) x = dist(rng);

    std::vector<float> partial(nthreads, 0.0f);
    double best_t = best_of_mt(nthreads, NTIMES, [&](unsigned tid, unsigned nt) {
        size_t begin, end;
        thread_range(N, tid, nt, begin, end);
        partial[tid] = dot(a.data() + begin, b.data() + begin, end - begin);
    });
    float result = 0.0f;
    for (auto p : partial) result += p;
    sink(result);

    double flops  = 2.0 * N;
    double bytes  = 2.0 * N * sizeof(float);
    double ai     = flops / bytes;
    double gflops = flops / best_t / 1e9;

    printf("dot product (MT)  N=%zuM floats, %u threads (best of %d)\n", N >> 20, nthreads, NTIMES);
    printf("  Time:        %.3f ms\n",        best_t * 1000.0);
    printf("  Arith. Int.: %.4f FLOP/byte\n", ai);
    printf("  Performance: %.4f GFLOP/s\n",  gflops);
    printf("DOTPROD_AI_MT=%.4f\n",     ai);
    printf("DOTPROD_GFLOPS_MT=%.4f\n", gflops);
    printf("MT_THREADS=%u\n", nthreads);
    return 0;
}
