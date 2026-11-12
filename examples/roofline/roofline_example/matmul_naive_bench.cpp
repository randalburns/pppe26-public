// Build: clang++ -std=c++17 -O3 -march=native -o matmul_naive_bench matmul_naive_bench.cpp
// Hand-written FP32 NxN matrix multiply (no BLAS, no AMX) in four variants —
// scalar mul+add, scalar FMA, SIMD mul+add, SIMD FMA — directly comparable to
// the four ceilings peak_flops.cpp measures, unlike matmul_bench.cpp's sgemm
// (which routes through the AMX coprocessor and isn't bound by them at all).
//
// Register-blocked micro-kernel: for a fixed tile of C, the full k reduction
// accumulates entirely in registers and is written back to memory only once,
// after the k loop, instead of round-tripping C and B through L1 on every
// single k step (an earlier unblocked version of this file did exactly
// that; measured against peak_flops.cpp's ceilings, both SIMD variants sat
// well below theirs — L1-bandwidth-bound, not FMA-issue-bound; blocking
// fixed that).
//
// Tile-shape history — three attempts to make SIMD FMA look as good as it
// should, each one tested and measured rather than assumed:
//  1. MR=4 rows x 1 vector column (4 accumulators). SIMD FMA (~37 GFLOP/s)
//     measured *slower* than SIMD mul+add (~47 GFLOP/s), despite fmla being
//     fewer instructions. First theory: the compiler scheduled the two
//     versions differently (mul+add loads all 4 A broadcasts before
//     consuming any of them; FMA interleaves each load with the fmla that
//     immediately consumes it). Tested directly with hand-written inline
//     assembly forcing the mul+add-style load order onto the FMA kernel —
//     made no difference at all (same ~38 GFLOP/s, bit-exact correct
//     output). Scheduling was not the cause.
//  2. MR=8 rows x 1 vector column (8 accumulators). Reasoned that a fused
//     fmla's full latency sits on the loop-carried critical path every k
//     step, while mul+add's critical path is just the add (the mul has no
//     cross-iteration dependency), so more independent chains should hide
//     that latency. It did narrow the gap (mul+add ~67, FMA ~66 — roughly
//     tied) but both landed well below their ceilings, and scalar FMA
//     actually got *worse* than at MR=4.
//  3. What was actually wrong with both 1 and 2: a 1-vector-column tile has
//     no B reuse across columns — every k step does ~1 FMA per load, a poor
//     ratio regardless of row count. Swept tile shapes directly (see the
//     conversation, not reproduced here) and found MR=4 rows x 4 vector
//     columns (16 accumulators, 2 loads -> 4 FMA per load-ish reuse) gets
//     SIMD FMA to ~124 GFLOP/s (93% of the 133.8 ceiling) against SIMD
//     mul+add's ~69 GFLOP/s — FMA finally wins by the large margin fewer
//     instructions should predict. The scalar kernels never had this
//     problem: their original 4x4 tile (16 scalar accumulators, 4 rows x 4
//     columns) already reused both A and B and lands at/above the scalar
//     ceilings, so they're unchanged.
// Lesson: the bottleneck was tile shape (load-to-FMA reuse ratio), not
// instruction scheduling and not raw accumulator count in isolation. Both
// of those were plausible-sounding theories that measurement disproved.
//
// Register budget: scalar tiles use 16 scalar accumulators (4x4) + a few
// operand registers, comfortably under AArch64's 32 FP/SIMD registers. SIMD
// tiles use 16 vector accumulators (4 rows x 4 vector columns) + operand
// registers — also verified by disassembly to stay spill-free (no str/stp
// of vector registers inside the k loop; only the one-time tile writeback
// after it).
//
// The "no FMA" variants insert an inline-asm register barrier between the
// multiply and the add so the compiler can't fuse them into a single fused
// multiply-add instruction. Verified by disassembly (not just by naming):
// scalar `a*b+c` fuses into `fmadd` by default without the barrier; NEON
// `vmulq_f32`+`vaddq_f32` don't auto-fuse into `fmla` either way, but the
// barrier is kept for both so the intent is explicit and robust.
//
// Arithmetic intensity = 2n^3 / (3n^2 x 4 bytes) = n/6, same convention as
// matmul_bench.cpp. n is chosen so the working set (3 x n^2 x 4 bytes = 3 MB)
// fits well inside the M5's L2/SLC, and so n is a multiple of the tile width
// (no cleanup-loop remainder to handle).

#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace std::chrono;

static const int N       = 512; // 3 x 512^2 x 4B = 3 MB, fits the SLC; multiple of 16
static const int NTIMES  = 5;
static const int MR      = 4;   // tile rows
static const int NR      = 4;   // scalar tile columns
static const int SIMD_NC = 4;   // SIMD tile columns, in 4-lane vectors (16 floats wide)

#define MULADD_NOFMA(dst, a, b) do { float _p = (a) * (b); asm volatile("" : "+w"(_p)); (dst) += _p; } while (0)
// Barrier after the fused result too: without it, Clang's SLP vectorizer
// (a straight-line-code optimization the loop-only vectorize(disable)
// pragma doesn't reach) merges these independent fmaf calls sharing one
// broadcast operand back into a single vector fmla — confirmed by
// disassembly. The barrier forces each result to materialize separately.
#define MULADD_FMA(dst, a, b) do { (dst) = __builtin_fmaf((a), (b), (dst)); asm volatile("" : "+w"(dst)); } while (0)

// Scalar tiles are 4 rows x 4 columns: 16 independent accumulators, each k
// step reusing 4 A loads and 4 B loads across all 16 of them.
__attribute__((noinline))
static void matmul_scalar(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int n) {
    asm volatile("" : : : "memory");
    for (int i = 0; i < n; i += MR) {
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
static void matmul_scalar_fma(const float* __restrict__ A, const float* __restrict__ B,
                               float* __restrict__ C, int n) {
    asm volatile("" : : : "memory");
    for (int i = 0; i < n; i += MR) {
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

// SIMD tiles are 4 rows x 4 vector columns (16 lanes wide, 16 accumulators):
// each k step loads 4 B vectors and 4 A broadcasts, reusing each of the 4 A
// broadcasts across all 4 B vectors — the reuse a 1-column tile lacked.
__attribute__((noinline))
static void matmul_simd(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C, int n) {
    asm volatile("" : : : "memory");
    for (int i = 0; i < n; i += MR) {
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
static void matmul_simd_fma(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int n) {
    asm volatile("" : : : "memory");
    for (int i = 0; i < n; i += MR) {
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

static double time_variant(void (*fn)(const float*, const float*, float*, int),
                            const float* A, const float* B, float* C, int n) {
    double best_t = 1e9;
    for (int t = 0; t < NTIMES; t++) {
        std::vector<float> c((size_t)n * n, 0.0f);
        auto t0 = high_resolution_clock::now();
        fn(A, B, c.data(), n);
        auto t1 = high_resolution_clock::now();
        double dt = duration<double>(t1 - t0).count();
        if (dt < best_t) best_t = dt;
        if (t == NTIMES - 1) std::copy(c.begin(), c.end(), C);
    }
    return best_t;
}

int main() {
    std::vector<float> A((size_t)N * N), B((size_t)N * N), C((size_t)N * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    struct Variant { const char* name; const char* tag; void (*fn)(const float*, const float*, float*, int); };
    Variant variants[] = {
        {"scalar",     "MATMUL_SCALAR",     matmul_scalar},
        {"scalar FMA", "MATMUL_SCALAR_FMA", matmul_scalar_fma},
        {"SIMD",       "MATMUL_SIMD",       matmul_simd},
        {"SIMD FMA",   "MATMUL_SIMD_FMA",   matmul_simd_fma},
    };

    double flops = 2.0 * N * N * N;
    double bytes = 3.0 * N * N * sizeof(float);
    double ai    = flops / bytes;

    printf("naive sgemm %dx%d, no BLAS/AMX (best of %d)\n", N, N, NTIMES);
    printf("  Arith. Int.: %.4f FLOP/byte\n", ai);
    float refC00 = 0.0f;
    for (auto& v : variants) {
        std::vector<float> C(N * N);
        double t = time_variant(v.fn, A.data(), B.data(), C.data(), N);
        double gflops = flops / t / 1e9;
        printf("  %-10s %8.3f ms   %8.2f GFLOP/s\n", v.name, t * 1000.0, gflops);
        printf("%s_GFLOPS=%.4f\n", v.tag, gflops);
        if (&v == &variants[0]) refC00 = C[0];
        else if (std::abs(C[0] - refC00) > 1e-2f * std::abs(refC00))
            fprintf(stderr, "WARNING: %s result mismatch (C[0]=%.4f vs %.4f)\n", v.name, C[0], refC00);
    }
    printf("MATMUL_AI=%.4f\n", ai);
    return 0;
}
