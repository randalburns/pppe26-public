// Build: clang++ -std=c++17 -O3 -march=native -o nbody_bench nbody_bench.cpp
// Measures FP32 N-body (naive O(N^2) pairwise, softened gravity) performance —
// the classic compute-bound kernel from the original roofline paper.
//
// FLOP count per interaction (accounting convention used here):
//   3 sub (dx,dy,dz) + 3 FMA for dx^2+dy^2+dz^2+eps^2 (6) + 4 for a
//   Newton-Raphson-refined reciprocal sqrt + 2 mul for invDist^3 + 1 mul for
//   mass + 3 FMA to accumulate the acceleration (6) = 22 FLOP/interaction.
//
// Arithmetic intensity: with all N particles resident in cache (16 bytes each:
// x,y,z,mass), only the first full sweep is a real DRAM read — the other N-1
// sweeps hit cache. AI = flops / (N * 16 bytes) grows with N: this kernel is
// deliberately compute-bound, the opposite end of the spectrum from dot_bench.

#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace std::chrono;

template <typename T>
static void sink(T const& v) { asm volatile("" : : "m"(v) : "memory"); }

static const int   N      = 4096;
static const int   NTIMES = 5;
static const float EPS2   = 1e-4f;

__attribute__((noinline))
static void nbody(const float* __restrict__ x, const float* __restrict__ y,
                   const float* __restrict__ z, const float* __restrict__ m,
                   float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
                   int n) {
    asm volatile("" : : : "memory");
    float32x4_t veps2 = vdupq_n_f32(EPS2);
    for (int i = 0; i < n; i++) {
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
    std::vector<float> x(N), y(N), z(N), m(N), ax(N), ay(N), az(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> mdist(0.1f, 1.0f);
    for (int i = 0; i < N; i++) { x[i]=dist(rng); y[i]=dist(rng); z[i]=dist(rng); m[i]=mdist(rng); }

    double best_t = 1e9;
    for (int k = 0; k < NTIMES; k++) {
        auto t0 = high_resolution_clock::now();
        nbody(x.data(), y.data(), z.data(), m.data(), ax.data(), ay.data(), az.data(), N);
        auto t1 = high_resolution_clock::now();
        double t = duration<double>(t1 - t0).count();
        if (t < best_t) best_t = t;
    }
    sink(ax[0]);

    double flops  = 22.0 * (double)N * (double)N;
    double bytes  = 16.0 * (double)N; // compulsory traffic: positions read once from DRAM
    double ai     = flops / bytes;
    double gflops = flops / best_t / 1e9;

    printf("nbody  N=%d particles (best of %d)\n", N, NTIMES);
    printf("  Time:        %.3f ms\n",        best_t * 1000.0);
    printf("  Arith. Int.: %.4f FLOP/byte\n", ai);
    printf("  Performance: %.4f GFLOP/s\n",  gflops);
    printf("NBODY_AI=%.4f\n",     ai);
    printf("NBODY_GFLOPS=%.4f\n", gflops);
    return 0;
}
