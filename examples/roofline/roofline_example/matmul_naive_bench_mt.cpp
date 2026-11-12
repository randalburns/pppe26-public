// Build: clang++ -std=c++17 -O3 -march=native -o matmul_naive_bench_mt matmul_naive_bench_mt.cpp
// Multicore companion to matmul_naive_bench.cpp: partitions C's row-tiles
// across threads (each thread owns a disjoint, contiguous range of 4-row
// tiles — same tile shape and kernel bodies as the single-thread file,
// verified there by disassembly). Deep in compute-bound territory
// (AI = 85.3), this is the kernel expected to scale best with cores of
// everything in this suite: no shared bandwidth ceiling to run into.

#include "mt_common.hpp"
#include <arm_neon.h>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

using namespace std::chrono;

template <typename T>
static void sink(T const& v) { asm volatile("" : : "m"(v) : "memory"); }

static const int N       = 512;
static const int NTIMES  = 5;
static const int MR      = 4;
static const int NR      = 4;
static const int SIMD_NC = 4;

#define MULADD_NOFMA(dst, a, b) do { float _p = (a) * (b); asm volatile("" : "+w"(_p)); (dst) += _p; } while (0)
#define MULADD_FMA(dst, a, b) do { (dst) = __builtin_fmaf((a), (b), (dst)); asm volatile("" : "+w"(dst)); } while (0)

// Identical kernel bodies to matmul_naive_bench.cpp, parameterized by the
// row-tile range [i_begin, i_end) a given thread owns.
__attribute__((noinline))
static void matmul_scalar_range(const float* __restrict__ A, const float* __restrict__ B,
                                 float* __restrict__ C, int n, int i_begin, int i_end) {
    asm volatile("" : : : "memory");
    for (int i = i_begin; i < i_end; i += MR) {
        for (int j = 0; j < n; j += NR) {
            float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
            float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
            float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
            float c30 = 0, c31 = 0, c32 = 0, c33 = 0;
            for (int k = 0; k < n; k++) {
                float a0 = A[(i + 0) * n + k], a1 = A[(i + 1) * n + k],
                      a2 = A[(i + 2) * n + k], a3 = A[(i + 3) * n + k];
                float b0 = B[k * n + j + 0], b1 = B[k * n + j + 1],
                      b2 = B[k * n + j + 2], b3 = B[k * n + j + 3];
                MULADD_NOFMA(c00, a0, b0); MULADD_NOFMA(c01, a0, b1); MULADD_NOFMA(c02, a0, b2); MULADD_NOFMA(c03, a0, b3);
                MULADD_NOFMA(c10, a1, b0); MULADD_NOFMA(c11, a1, b1); MULADD_NOFMA(c12, a1, b2); MULADD_NOFMA(c13, a1, b3);
                MULADD_NOFMA(c20, a2, b0); MULADD_NOFMA(c21, a2, b1); MULADD_NOFMA(c22, a2, b2); MULADD_NOFMA(c23, a2, b3);
                MULADD_NOFMA(c30, a3, b0); MULADD_NOFMA(c31, a3, b1); MULADD_NOFMA(c32, a3, b2); MULADD_NOFMA(c33, a3, b3);
            }
            C[(i+0)*n+j+0]=c00; C[(i+0)*n+j+1]=c01; C[(i+0)*n+j+2]=c02; C[(i+0)*n+j+3]=c03;
            C[(i+1)*n+j+0]=c10; C[(i+1)*n+j+1]=c11; C[(i+1)*n+j+2]=c12; C[(i+1)*n+j+3]=c13;
            C[(i+2)*n+j+0]=c20; C[(i+2)*n+j+1]=c21; C[(i+2)*n+j+2]=c22; C[(i+2)*n+j+3]=c23;
            C[(i+3)*n+j+0]=c30; C[(i+3)*n+j+1]=c31; C[(i+3)*n+j+2]=c32; C[(i+3)*n+j+3]=c33;
        }
    }
}

__attribute__((noinline))
static void matmul_scalar_fma_range(const float* __restrict__ A, const float* __restrict__ B,
                                     float* __restrict__ C, int n, int i_begin, int i_end) {
    asm volatile("" : : : "memory");
    for (int i = i_begin; i < i_end; i += MR) {
        for (int j = 0; j < n; j += NR) {
            float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
            float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
            float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
            float c30 = 0, c31 = 0, c32 = 0, c33 = 0;
            for (int k = 0; k < n; k++) {
                float a0 = A[(i + 0) * n + k], a1 = A[(i + 1) * n + k],
                      a2 = A[(i + 2) * n + k], a3 = A[(i + 3) * n + k];
                float b0 = B[k * n + j + 0], b1 = B[k * n + j + 1],
                      b2 = B[k * n + j + 2], b3 = B[k * n + j + 3];
                MULADD_FMA(c00, a0, b0); MULADD_FMA(c01, a0, b1); MULADD_FMA(c02, a0, b2); MULADD_FMA(c03, a0, b3);
                MULADD_FMA(c10, a1, b0); MULADD_FMA(c11, a1, b1); MULADD_FMA(c12, a1, b2); MULADD_FMA(c13, a1, b3);
                MULADD_FMA(c20, a2, b0); MULADD_FMA(c21, a2, b1); MULADD_FMA(c22, a2, b2); MULADD_FMA(c23, a2, b3);
                MULADD_FMA(c30, a3, b0); MULADD_FMA(c31, a3, b1); MULADD_FMA(c32, a3, b2); MULADD_FMA(c33, a3, b3);
            }
            C[(i+0)*n+j+0]=c00; C[(i+0)*n+j+1]=c01; C[(i+0)*n+j+2]=c02; C[(i+0)*n+j+3]=c03;
            C[(i+1)*n+j+0]=c10; C[(i+1)*n+j+1]=c11; C[(i+1)*n+j+2]=c12; C[(i+1)*n+j+3]=c13;
            C[(i+2)*n+j+0]=c20; C[(i+2)*n+j+1]=c21; C[(i+2)*n+j+2]=c22; C[(i+2)*n+j+3]=c23;
            C[(i+3)*n+j+0]=c30; C[(i+3)*n+j+1]=c31; C[(i+3)*n+j+2]=c32; C[(i+3)*n+j+3]=c33;
        }
    }
}

__attribute__((noinline))
static void matmul_simd_range(const float* __restrict__ A, const float* __restrict__ B,
                               float* __restrict__ C, int n, int i_begin, int i_end) {
    asm volatile("" : : : "memory");
    for (int i = i_begin; i < i_end; i += MR) {
        for (int j = 0; j < n; j += SIMD_NC * 4) {
            float32x4_t c00 = vdupq_n_f32(0.0f), c01 = vdupq_n_f32(0.0f), c02 = vdupq_n_f32(0.0f), c03 = vdupq_n_f32(0.0f);
            float32x4_t c10 = vdupq_n_f32(0.0f), c11 = vdupq_n_f32(0.0f), c12 = vdupq_n_f32(0.0f), c13 = vdupq_n_f32(0.0f);
            float32x4_t c20 = vdupq_n_f32(0.0f), c21 = vdupq_n_f32(0.0f), c22 = vdupq_n_f32(0.0f), c23 = vdupq_n_f32(0.0f);
            float32x4_t c30 = vdupq_n_f32(0.0f), c31 = vdupq_n_f32(0.0f), c32 = vdupq_n_f32(0.0f), c33 = vdupq_n_f32(0.0f);
            for (int k = 0; k < n; k++) {
                float32x4_t b0 = vld1q_f32(&B[k * n + j + 0]),  b1 = vld1q_f32(&B[k * n + j + 4]),
                            b2 = vld1q_f32(&B[k * n + j + 8]),  b3 = vld1q_f32(&B[k * n + j + 12]);
                float32x4_t a0 = vdupq_n_f32(A[(i + 0) * n + k]), a1 = vdupq_n_f32(A[(i + 1) * n + k]),
                            a2 = vdupq_n_f32(A[(i + 2) * n + k]), a3 = vdupq_n_f32(A[(i + 3) * n + k]);
                float32x4_t p00 = vmulq_f32(a0, b0); asm volatile("" : "+w"(p00));
                float32x4_t p01 = vmulq_f32(a0, b1); asm volatile("" : "+w"(p01));
                float32x4_t p02 = vmulq_f32(a0, b2); asm volatile("" : "+w"(p02));
                float32x4_t p03 = vmulq_f32(a0, b3); asm volatile("" : "+w"(p03));
                float32x4_t p10 = vmulq_f32(a1, b0); asm volatile("" : "+w"(p10));
                float32x4_t p11 = vmulq_f32(a1, b1); asm volatile("" : "+w"(p11));
                float32x4_t p12 = vmulq_f32(a1, b2); asm volatile("" : "+w"(p12));
                float32x4_t p13 = vmulq_f32(a1, b3); asm volatile("" : "+w"(p13));
                float32x4_t p20 = vmulq_f32(a2, b0); asm volatile("" : "+w"(p20));
                float32x4_t p21 = vmulq_f32(a2, b1); asm volatile("" : "+w"(p21));
                float32x4_t p22 = vmulq_f32(a2, b2); asm volatile("" : "+w"(p22));
                float32x4_t p23 = vmulq_f32(a2, b3); asm volatile("" : "+w"(p23));
                float32x4_t p30 = vmulq_f32(a3, b0); asm volatile("" : "+w"(p30));
                float32x4_t p31 = vmulq_f32(a3, b1); asm volatile("" : "+w"(p31));
                float32x4_t p32 = vmulq_f32(a3, b2); asm volatile("" : "+w"(p32));
                float32x4_t p33 = vmulq_f32(a3, b3); asm volatile("" : "+w"(p33));
                c00 = vaddq_f32(c00, p00); c01 = vaddq_f32(c01, p01); c02 = vaddq_f32(c02, p02); c03 = vaddq_f32(c03, p03);
                c10 = vaddq_f32(c10, p10); c11 = vaddq_f32(c11, p11); c12 = vaddq_f32(c12, p12); c13 = vaddq_f32(c13, p13);
                c20 = vaddq_f32(c20, p20); c21 = vaddq_f32(c21, p21); c22 = vaddq_f32(c22, p22); c23 = vaddq_f32(c23, p23);
                c30 = vaddq_f32(c30, p30); c31 = vaddq_f32(c31, p31); c32 = vaddq_f32(c32, p32); c33 = vaddq_f32(c33, p33);
            }
            vst1q_f32(&C[(i+0)*n+j+0], c00); vst1q_f32(&C[(i+0)*n+j+4], c01); vst1q_f32(&C[(i+0)*n+j+8], c02); vst1q_f32(&C[(i+0)*n+j+12], c03);
            vst1q_f32(&C[(i+1)*n+j+0], c10); vst1q_f32(&C[(i+1)*n+j+4], c11); vst1q_f32(&C[(i+1)*n+j+8], c12); vst1q_f32(&C[(i+1)*n+j+12], c13);
            vst1q_f32(&C[(i+2)*n+j+0], c20); vst1q_f32(&C[(i+2)*n+j+4], c21); vst1q_f32(&C[(i+2)*n+j+8], c22); vst1q_f32(&C[(i+2)*n+j+12], c23);
            vst1q_f32(&C[(i+3)*n+j+0], c30); vst1q_f32(&C[(i+3)*n+j+4], c31); vst1q_f32(&C[(i+3)*n+j+8], c32); vst1q_f32(&C[(i+3)*n+j+12], c33);
        }
    }
}

__attribute__((noinline))
static void matmul_simd_fma_range(const float* __restrict__ A, const float* __restrict__ B,
                                   float* __restrict__ C, int n, int i_begin, int i_end) {
    asm volatile("" : : : "memory");
    for (int i = i_begin; i < i_end; i += MR) {
        for (int j = 0; j < n; j += SIMD_NC * 4) {
            float32x4_t c00 = vdupq_n_f32(0.0f), c01 = vdupq_n_f32(0.0f), c02 = vdupq_n_f32(0.0f), c03 = vdupq_n_f32(0.0f);
            float32x4_t c10 = vdupq_n_f32(0.0f), c11 = vdupq_n_f32(0.0f), c12 = vdupq_n_f32(0.0f), c13 = vdupq_n_f32(0.0f);
            float32x4_t c20 = vdupq_n_f32(0.0f), c21 = vdupq_n_f32(0.0f), c22 = vdupq_n_f32(0.0f), c23 = vdupq_n_f32(0.0f);
            float32x4_t c30 = vdupq_n_f32(0.0f), c31 = vdupq_n_f32(0.0f), c32 = vdupq_n_f32(0.0f), c33 = vdupq_n_f32(0.0f);
            for (int k = 0; k < n; k++) {
                float32x4_t b0 = vld1q_f32(&B[k * n + j + 0]),  b1 = vld1q_f32(&B[k * n + j + 4]),
                            b2 = vld1q_f32(&B[k * n + j + 8]),  b3 = vld1q_f32(&B[k * n + j + 12]);
                float32x4_t a0 = vdupq_n_f32(A[(i + 0) * n + k]), a1 = vdupq_n_f32(A[(i + 1) * n + k]),
                            a2 = vdupq_n_f32(A[(i + 2) * n + k]), a3 = vdupq_n_f32(A[(i + 3) * n + k]);
                c00 = vfmaq_f32(c00, a0, b0); c01 = vfmaq_f32(c01, a0, b1); c02 = vfmaq_f32(c02, a0, b2); c03 = vfmaq_f32(c03, a0, b3);
                c10 = vfmaq_f32(c10, a1, b0); c11 = vfmaq_f32(c11, a1, b1); c12 = vfmaq_f32(c12, a1, b2); c13 = vfmaq_f32(c13, a1, b3);
                c20 = vfmaq_f32(c20, a2, b0); c21 = vfmaq_f32(c21, a2, b1); c22 = vfmaq_f32(c22, a2, b2); c23 = vfmaq_f32(c23, a2, b3);
                c30 = vfmaq_f32(c30, a3, b0); c31 = vfmaq_f32(c31, a3, b1); c32 = vfmaq_f32(c32, a3, b2); c33 = vfmaq_f32(c33, a3, b3);
            }
            vst1q_f32(&C[(i+0)*n+j+0], c00); vst1q_f32(&C[(i+0)*n+j+4], c01); vst1q_f32(&C[(i+0)*n+j+8], c02); vst1q_f32(&C[(i+0)*n+j+12], c03);
            vst1q_f32(&C[(i+1)*n+j+0], c10); vst1q_f32(&C[(i+1)*n+j+4], c11); vst1q_f32(&C[(i+1)*n+j+8], c12); vst1q_f32(&C[(i+1)*n+j+12], c13);
            vst1q_f32(&C[(i+2)*n+j+0], c20); vst1q_f32(&C[(i+2)*n+j+4], c21); vst1q_f32(&C[(i+2)*n+j+8], c22); vst1q_f32(&C[(i+2)*n+j+12], c23);
            vst1q_f32(&C[(i+3)*n+j+0], c30); vst1q_f32(&C[(i+3)*n+j+4], c31); vst1q_f32(&C[(i+3)*n+j+8], c32); vst1q_f32(&C[(i+3)*n+j+12], c33);
        }
    }
}

static double time_variant_mt(void (*fn)(const float*, const float*, float*, int, int, int),
                               unsigned nthreads, const float* A, const float* B, float* C, int n) {
    int ntiles = n / MR;
    return best_of_mt(nthreads, NTIMES, [&](unsigned tid, unsigned nt) {
        size_t tbegin, tend;
        thread_range((size_t)ntiles, tid, nt, tbegin, tend);
        fn(A, B, C, n, (int)tbegin * MR, (int)tend * MR);
    });
}

int main() {
    unsigned nthreads = std::thread::hardware_concurrency();
    std::vector<float> A((size_t)N * N), B((size_t)N * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    struct Variant {
        const char* name; const char* tag;
        void (*fn)(const float*, const float*, float*, int, int, int);
    };
    Variant variants[] = {
        {"scalar",     "MATMUL_SCALAR_MT",     matmul_scalar_range},
        {"scalar FMA", "MATMUL_SCALAR_FMA_MT", matmul_scalar_fma_range},
        {"SIMD",       "MATMUL_SIMD_MT",       matmul_simd_range},
        {"SIMD FMA",   "MATMUL_SIMD_FMA_MT",   matmul_simd_fma_range},
    };

    double flops = 2.0 * N * N * N;
    double bytes = 3.0 * N * N * sizeof(float);
    double ai    = flops / bytes;

    printf("naive sgemm (MT) %dx%d, no BLAS/AMX, %u threads (best of %d)\n", N, N, nthreads, NTIMES);
    printf("  Arith. Int.: %.4f FLOP/byte\n", ai);
    for (auto& v : variants) {
        std::vector<float> C((size_t)N * N, 0.0f);
        double t = time_variant_mt(v.fn, nthreads, A.data(), B.data(), C.data(), N);
        double gflops = flops / t / 1e9;
        printf("  %-10s %8.3f ms   %8.2f GFLOP/s\n", v.name, t * 1000.0, gflops);
        printf("%s_GFLOPS=%.4f\n", v.tag, gflops);
        sink(C[0]);
    }
    printf("MATMUL_AI_MT=%.4f\n", ai);
    printf("MT_THREADS=%u\n", nthreads);
    return 0;
}
