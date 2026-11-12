// Build: clang++ -std=c++17 -O3 -march=native -o nbody_bench_mt nbody_bench_mt.cpp
// Multicore companion to nbody_bench.cpp: partitions the outer particle loop
// across threads (each particle's acceleration is independent of every
// other particle's *output*, though all threads read the full shared
// position/mass arrays) — the textbook embarrassingly-parallel decomposition
// for N-body, same one GPUs use. Being deep in compute-bound territory
// (AI in the thousands), this should scale close to linearly with cores,
// unlike any of the memory-bound kernels above.

#include "mt_common.hpp"
#include <arm_neon.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

template <typename T>
static void sink(T const& v) { asm volatile("" : : "m"(v) : "memory"); }

static const int   N      = 4096; // same as nbody_bench.cpp
static const int   NTIMES = 5;
static const float EPS2   = 1e-4f;

// Identical kernel to nbody_bench.cpp's nbody(), operating on a row range.
__attribute__((noinline))
static void nbody_range(const float* __restrict__ x, const float* __restrict__ y,
                         const float* __restrict__ z, const float* __restrict__ m,
                         float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
                         int n, int i_begin, int i_end) {
    asm volatile("" : : : "memory");
    float32x4_t veps2 = vdupq_n_f32(EPS2);
    for (int i = i_begin; i < i_end; i++) {
        float32x4_t xi = vdupq_n_f32(x[i]), yi = vdupq_n_f32(y[i]), zi = vdupq_n_f32(z[i]);
        float32x4_t axi = vdupq_n_f32(0.0f), ayi = vdupq_n_f32(0.0f), azi = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            float32x4_t dx = vsubq_f32(vld1q_f32(x + j), xi);
            float32x4_t dy = vsubq_f32(vld1q_f32(y + j), yi);
            float32x4_t dz = vsubq_f32(vld1q_f32(z + j), zi);
            float32x4_t d2 = vfmaq_f32(vfmaq_f32(vfmaq_f32(veps2, dx, dx), dy, dy), dz, dz);
            float32x4_t r  = vrsqrteq_f32(d2);
            r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(d2, r), r));
            float32x4_t r3 = vmulq_f32(vmulq_f32(r, r), r);
            float32x4_t s  = vmulq_f32(vld1q_f32(m + j), r3);
            axi = vfmaq_f32(axi, dx, s);
            ayi = vfmaq_f32(ayi, dy, s);
            azi = vfmaq_f32(azi, dz, s);
        }
        float sx = vaddvq_f32(axi), sy = vaddvq_f32(ayi), sz = vaddvq_f32(azi);
        for (; j < n; j++) {
            float dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
            float d2 = dx * dx + dy * dy + dz * dz + EPS2;
            float invd = 1.0f / std::sqrt(d2);
            float invd3 = invd * invd * invd;
            float s = m[j] * invd3;
            sx += dx * s; sy += dy * s; sz += dz * s;
        }
        ax[i] = sx; ay[i] = sy; az[i] = sz;
    }
}

int main() {
    unsigned nthreads = std::thread::hardware_concurrency();
    std::vector<float> x(N), y(N), z(N), m(N), ax(N), ay(N), az(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> mdist(0.1f, 1.0f);
    for (int i = 0; i < N; i++) { x[i]=dist(rng); y[i]=dist(rng); z[i]=dist(rng); m[i]=mdist(rng); }

    double best_t = best_of_mt(nthreads, NTIMES, [&](unsigned tid, unsigned nt) {
        size_t begin, end;
        thread_range(N, tid, nt, begin, end);
        nbody_range(x.data(), y.data(), z.data(), m.data(), ax.data(), ay.data(), az.data(),
                    N, (int)begin, (int)end);
    });
    sink(ax[0]);

    double flops  = 22.0 * (double)N * (double)N;
    double bytes  = 16.0 * (double)N;
    double ai     = flops / bytes;
    double gflops = flops / best_t / 1e9;

    printf("nbody (MT)  N=%d particles, %u threads (best of %d)\n", N, nthreads, NTIMES);
    printf("  Time:        %.3f ms\n",        best_t * 1000.0);
    printf("  Arith. Int.: %.4f FLOP/byte\n", ai);
    printf("  Performance: %.4f GFLOP/s\n",  gflops);
    printf("NBODY_AI_MT=%.4f\n",     ai);
    printf("NBODY_GFLOPS_MT=%.4f\n", gflops);
    printf("MT_THREADS=%u\n", nthreads);
    return 0;
}
