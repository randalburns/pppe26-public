// Build: clang++ -std=c++17 -O3 -march=native -o peak_flops_mt peak_flops_mt.cpp
// Multicore companion to peak_flops.cpp: every core runs the same
// register-only FMA/mul kernel concurrently, no shared data at all, so
// unlike stream_mt.cpp this should scale close to linearly with thread
// count — compute throughput isn't a shared resource the way DRAM
// bandwidth is. How close it actually gets says something about thermal/
// power headroom and P-core/E-core heterogeneity (this chip is 4 P-cores +
// 6 E-cores, not 10 identical cores).
//
// Kernel body is copied verbatim from peak_flops.cpp's run_simd/run_scalar
// functions (same 30-accumulator design to saturate FMA units and dodge
// register spilling) — only the "run on N threads, sum the throughput"
// wrapper is new.

#include "mt_common.hpp"
#include <arm_neon.h>
#include <cstdio>
#include <thread>

template <typename T>
static void sink(T const& v) { asm volatile("" : : "m"(v) : "memory"); }

static const long long ITERS   = 1LL << 27;
static const int       NTIMES  = 5;

#define DECL_SIMD(b, c) \
    float32x4_t a0 =vdupq_n_f32(0.1f), a1 =vdupq_n_f32(0.2f), \
                a2 =vdupq_n_f32(0.3f), a3 =vdupq_n_f32(0.4f), \
                a4 =vdupq_n_f32(0.5f), a5 =vdupq_n_f32(0.6f), \
                a6 =vdupq_n_f32(0.7f), a7 =vdupq_n_f32(0.8f), \
                a8 =vdupq_n_f32(0.11f),a9 =vdupq_n_f32(0.12f),\
                a10=vdupq_n_f32(0.13f),a11=vdupq_n_f32(0.14f),\
                a12=vdupq_n_f32(0.15f),a13=vdupq_n_f32(0.16f),\
                a14=vdupq_n_f32(0.17f),a15=vdupq_n_f32(0.18f),\
                a16=vdupq_n_f32(0.19f),a17=vdupq_n_f32(0.20f),\
                a18=vdupq_n_f32(0.21f),a19=vdupq_n_f32(0.22f),\
                a20=vdupq_n_f32(0.23f),a21=vdupq_n_f32(0.24f),\
                a22=vdupq_n_f32(0.25f),a23=vdupq_n_f32(0.26f),\
                a24=vdupq_n_f32(0.27f),a25=vdupq_n_f32(0.28f),\
                a26=vdupq_n_f32(0.29f),a27=vdupq_n_f32(0.30f),\
                a28=vdupq_n_f32(0.31f),a29=vdupq_n_f32(0.32f);

#define SINK_SIMD \
    sink(a0);sink(a1);sink(a2);sink(a3);sink(a4);sink(a5);sink(a6);sink(a7); \
    sink(a8);sink(a9);sink(a10);sink(a11);sink(a12);sink(a13);sink(a14);sink(a15); \
    sink(a16);sink(a17);sink(a18);sink(a19);sink(a20);sink(a21);sink(a22);sink(a23); \
    sink(a24);sink(a25);sink(a26);sink(a27);sink(a28);sink(a29);

__attribute__((noinline))
static void run_simd_fma(long long iters) {
    float32x4_t b = vdupq_n_f32(1.0001f), c = vdupq_n_f32(1.0002f);
    DECL_SIMD(b, c)
    for (long long i = 0; i < iters; i++) {
        a0 =vfmaq_f32(a0, b,c); a1 =vfmaq_f32(a1, b,c); a2 =vfmaq_f32(a2, b,c); a3 =vfmaq_f32(a3, b,c);
        a4 =vfmaq_f32(a4, b,c); a5 =vfmaq_f32(a5, b,c); a6 =vfmaq_f32(a6, b,c); a7 =vfmaq_f32(a7, b,c);
        a8 =vfmaq_f32(a8, b,c); a9 =vfmaq_f32(a9, b,c); a10=vfmaq_f32(a10,b,c); a11=vfmaq_f32(a11,b,c);
        a12=vfmaq_f32(a12,b,c); a13=vfmaq_f32(a13,b,c); a14=vfmaq_f32(a14,b,c); a15=vfmaq_f32(a15,b,c);
        a16=vfmaq_f32(a16,b,c); a17=vfmaq_f32(a17,b,c); a18=vfmaq_f32(a18,b,c); a19=vfmaq_f32(a19,b,c);
        a20=vfmaq_f32(a20,b,c); a21=vfmaq_f32(a21,b,c); a22=vfmaq_f32(a22,b,c); a23=vfmaq_f32(a23,b,c);
        a24=vfmaq_f32(a24,b,c); a25=vfmaq_f32(a25,b,c); a26=vfmaq_f32(a26,b,c); a27=vfmaq_f32(a27,b,c);
        a28=vfmaq_f32(a28,b,c); a29=vfmaq_f32(a29,b,c);
    }
    SINK_SIMD
}

__attribute__((noinline))
static void run_simd_nofma(long long iters) {
    float32x4_t b = vdupq_n_f32(1.0001f), c = vdupq_n_f32(1.0002f);
    DECL_SIMD(b, c)
    for (long long i = 0; i < iters; i++) {
        a0 =vmulq_f32(a0, b); a1 =vmulq_f32(a1, b); a2 =vmulq_f32(a2, b); a3 =vmulq_f32(a3, b);
        a4 =vmulq_f32(a4, b); a5 =vmulq_f32(a5, b); a6 =vmulq_f32(a6, b); a7 =vmulq_f32(a7, b);
        a8 =vmulq_f32(a8, b); a9 =vmulq_f32(a9, b); a10=vmulq_f32(a10,b); a11=vmulq_f32(a11,b);
        a12=vmulq_f32(a12,b); a13=vmulq_f32(a13,b); a14=vmulq_f32(a14,b); a15=vmulq_f32(a15,b);
        a16=vmulq_f32(a16,b); a17=vmulq_f32(a17,b); a18=vmulq_f32(a18,b); a19=vmulq_f32(a19,b);
        a20=vmulq_f32(a20,b); a21=vmulq_f32(a21,b); a22=vmulq_f32(a22,b); a23=vmulq_f32(a23,b);
        a24=vmulq_f32(a24,b); a25=vmulq_f32(a25,b); a26=vmulq_f32(a26,b); a27=vmulq_f32(a27,b);
        a28=vmulq_f32(a28,b); a29=vmulq_f32(a29,b);
    }
    SINK_SIMD
}

__attribute__((noinline))
static void run_scalar_nofma(long long iters) {
    float b=1.0001f;
    float a0=.1f,a1=.2f,a2=.3f,a3=.4f,a4=.5f,a5=.6f,a6=.7f,a7=.8f,
          a8=.11f,a9=.12f,a10=.13f,a11=.14f,a12=.15f,a13=.16f,a14=.17f,a15=.18f,
          a16=.19f,a17=.20f,a18=.21f,a19=.22f,a20=.23f,a21=.24f,a22=.25f,a23=.26f,
          a24=.27f,a25=.28f,a26=.29f,a27=.30f,a28=.31f,a29=.32f;
    for (long long i = 0; i < iters; i++) {
        a0*=b; a1*=b; a2*=b; a3*=b; a4*=b; a5*=b; a6*=b; a7*=b;
        a8*=b; a9*=b; a10*=b; a11*=b; a12*=b; a13*=b; a14*=b; a15*=b;
        a16*=b; a17*=b; a18*=b; a19*=b; a20*=b; a21*=b; a22*=b; a23*=b;
        a24*=b; a25*=b; a26*=b; a27*=b; a28*=b; a29*=b;
    }
    sink(a0);sink(a1);sink(a2);sink(a3);sink(a4);sink(a5);sink(a6);sink(a7);
    sink(a8);sink(a9);sink(a10);sink(a11);sink(a12);sink(a13);sink(a14);sink(a15);
    sink(a16);sink(a17);sink(a18);sink(a19);sink(a20);sink(a21);sink(a22);sink(a23);
    sink(a24);sink(a25);sink(a26);sink(a27);sink(a28);sink(a29);
}

__attribute__((noinline))
static void run_scalar_fma(long long iters) {
    float b=1.0001f,c=1.0002f;
    float a0=.1f,a1=.2f,a2=.3f,a3=.4f,a4=.5f,a5=.6f,a6=.7f,a7=.8f,
          a8=.11f,a9=.12f,a10=.13f,a11=.14f,a12=.15f,a13=.16f,a14=.17f,a15=.18f,
          a16=.19f,a17=.20f,a18=.21f,a19=.22f,a20=.23f,a21=.24f,a22=.25f,a23=.26f,
          a24=.27f,a25=.28f,a26=.29f,a27=.30f,a28=.31f,a29=.32f;
    for (long long i = 0; i < iters; i++) {
        a0=__builtin_fmaf(b,c,a0);   a1=__builtin_fmaf(b,c,a1);
        a2=__builtin_fmaf(b,c,a2);   a3=__builtin_fmaf(b,c,a3);
        a4=__builtin_fmaf(b,c,a4);   a5=__builtin_fmaf(b,c,a5);
        a6=__builtin_fmaf(b,c,a6);   a7=__builtin_fmaf(b,c,a7);
        a8=__builtin_fmaf(b,c,a8);   a9=__builtin_fmaf(b,c,a9);
        a10=__builtin_fmaf(b,c,a10); a11=__builtin_fmaf(b,c,a11);
        a12=__builtin_fmaf(b,c,a12); a13=__builtin_fmaf(b,c,a13);
        a14=__builtin_fmaf(b,c,a14); a15=__builtin_fmaf(b,c,a15);
        a16=__builtin_fmaf(b,c,a16); a17=__builtin_fmaf(b,c,a17);
        a18=__builtin_fmaf(b,c,a18); a19=__builtin_fmaf(b,c,a19);
        a20=__builtin_fmaf(b,c,a20); a21=__builtin_fmaf(b,c,a21);
        a22=__builtin_fmaf(b,c,a22); a23=__builtin_fmaf(b,c,a23);
        a24=__builtin_fmaf(b,c,a24); a25=__builtin_fmaf(b,c,a25);
        a26=__builtin_fmaf(b,c,a26); a27=__builtin_fmaf(b,c,a27);
        a28=__builtin_fmaf(b,c,a28); a29=__builtin_fmaf(b,c,a29);
    }
    sink(a0);sink(a1);sink(a2);sink(a3);sink(a4);sink(a5);sink(a6);sink(a7);
    sink(a8);sink(a9);sink(a10);sink(a11);sink(a12);sink(a13);sink(a14);sink(a15);
    sink(a16);sink(a17);sink(a18);sink(a19);sink(a20);sink(a21);sink(a22);sink(a23);
    sink(a24);sink(a25);sink(a26);sink(a27);sink(a28);sink(a29);
}

static double aggregate_gflops(unsigned nthreads, void (*fn)(long long), double flop_per_iter) {
    fn(1LL << 24); // per-thread warmup (small, on thread 0's core only, but fine)
    double best_t = best_of_mt(nthreads, NTIMES, [&](unsigned, unsigned) { fn(ITERS); });
    double total_flops = flop_per_iter * (double)ITERS * (double)nthreads;
    return total_flops / best_t / 1e9;
}

int main() {
    unsigned nthreads = std::thread::hardware_concurrency();
    printf("Threads: %u\n", nthreads);

    double simd_fma_gf     = aggregate_gflops(nthreads, run_simd_fma,     240.0);
    double simd_nofma_gf   = aggregate_gflops(nthreads, run_simd_nofma,   120.0);
    double scalar_fma_gf   = aggregate_gflops(nthreads, run_scalar_fma,    60.0);
    double scalar_nofma_gf = aggregate_gflops(nthreads, run_scalar_nofma,  30.0);

    printf("FP32 SIMD FMA   (NEON x4), aggregate: %8.2f GFLOP/s\n", simd_fma_gf);
    printf("FP32 SIMD no-FMA, aggregate         : %8.2f GFLOP/s\n", simd_nofma_gf);
    printf("FP32 scalar FMA, aggregate          : %8.2f GFLOP/s\n", scalar_fma_gf);
    printf("FP32 scalar no-FMA, aggregate        : %8.2f GFLOP/s\n", scalar_nofma_gf);
    printf("PEAK_GFLOPS_SIMD_FMA_MT=%.2f\n",    simd_fma_gf);
    printf("PEAK_GFLOPS_SIMD_NOFMA_MT=%.2f\n",  simd_nofma_gf);
    printf("PEAK_GFLOPS_SCALAR_FMA_MT=%.2f\n",  scalar_fma_gf);
    printf("PEAK_GFLOPS_SCALAR_NOFMA_MT=%.2f\n",scalar_nofma_gf);
    printf("MT_THREADS=%u\n", nthreads);
    return 0;
}
