// Build: clang++ -std=c++17 -O3 -march=native -o dot_bench dot_bench.cpp
// Measures FP32 dot product performance.
//
// Arithmetic intensity = 2 FLOP (mul+add) / 8 bytes (read a[i], b[i])
//                      = 0.25 FLOP/byte  →  memory-bound for any bandwidth ceiling
//
// Arrays are sized well beyond the M5 SLC (12 MB) so reads come from DRAM.

#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace std::chrono;

template <typename T>
static void sink(T const& v) { asm volatile("" : : "m"(v) : "memory"); }

static const size_t N      = 32ULL * 1024 * 1024; // 32M floats = 256 MB total (both arrays)
static const int    NTIMES = 5;

// 4 independent NEON accumulators so the FMA chain latency doesn't bottleneck
// the loop ahead of DRAM bandwidth; horizontal reduction happens once at the end.
// The leading asm barrier stops the compiler from proving this call pure and
// CSE-ing the repeated identical calls in the timing loop below.
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
    std::vector<float> a(N), b(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : a) x = dist(rng);
    for (auto& x : b) x = dist(rng);

    double best_t = 1e9;
    float  result = 0.0f;
    for (int k = 0; k < NTIMES; k++) {
        auto t0 = high_resolution_clock::now();
        result = dot(a.data(), b.data(), N);
        auto t1 = high_resolution_clock::now();
        double t = duration<double>(t1 - t0).count();
        if (t < best_t) best_t = t;
    }
    sink(result);

    double flops  = 2.0 * N;               // N multiplies + N adds
    double bytes  = 2.0 * N * sizeof(float); // read a[] and b[] once each
    double ai     = flops / bytes;          // = 0.25 FLOP/byte
    double gflops = flops / best_t / 1e9;

    printf("dot product  N=%zuM floats (best of %d)\n", N >> 20, NTIMES);
    printf("  Time:        %.3f ms\n",        best_t * 1000.0);
    printf("  Arith. Int.: %.4f FLOP/byte\n", ai);
    printf("  Performance: %.4f GFLOP/s\n",  gflops);
    printf("DOTPROD_AI=%.4f\n",     ai);
    printf("DOTPROD_GFLOPS=%.4f\n", gflops);
    return 0;
}
