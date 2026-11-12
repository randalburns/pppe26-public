// Build: clang++ -std=c++17 -O3 -march=native -o stencil_bench_mt stencil_bench_mt.cpp
// Multicore companion to stencil_bench.cpp: partitions the outer i (x-plane)
// loop across threads — the standard spatial domain decomposition for
// stencil codes. Safe without synchronization: each thread only writes its
// own disjoint range of `out` planes; `in` is read-only for the whole
// parallel region, including the one-plane overlap each thread reads at its
// boundary (i-1/i+1 from a neighboring thread's range).
//
// stencil_bench.cpp measured *more* effective bandwidth than its naive AI
// predicted, from cache-line reuse a byte-counting model doesn't capture.
// Multicore is the natural next question for exactly that kind of kernel:
// does the reuse that helped a single core also let this scale with cores,
// or is it ultimately capped the same way dot_bench_mt.cpp was?

#include "mt_common.hpp"
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

template <typename T>
static void sink(T const& v) { asm volatile("" : : "m"(v) : "memory"); }

static const int NX = 256, NY = 256, NZ = 256; // same grid as stencil_bench.cpp
static const int NTIMES = 5;

__attribute__((noinline))
static void stencil_range(const float* __restrict__ in, float* __restrict__ out,
                           int nx, int ny, int nz, float c0, float c1,
                           int i_begin, int i_end) {
    asm volatile("" : : : "memory");
    size_t nynz = (size_t)ny * nz;
    for (int i = i_begin; i < i_end; i++) {
        for (int j = 1; j < ny - 1; j++) {
            size_t base = (size_t)i * nynz + (size_t)j * nz;
            for (int k = 1; k < nz - 1; k++) {
                size_t idx = base + k;
                out[idx] = c0 * in[idx] + c1 * (in[idx - 1] + in[idx + 1] +
                                                 in[idx - nz] + in[idx + nz] +
                                                 in[idx - nynz] + in[idx + nynz]);
            }
        }
    }
}

int main() {
    unsigned nthreads = std::thread::hardware_concurrency();
    size_t n = (size_t)NX * NY * NZ;
    std::vector<float> a(n), b(n, 0.0f);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : a) v = dist(rng);

    double best_t = best_of_mt(nthreads, NTIMES, [&](unsigned tid, unsigned nt) {
        // Interior planes only [1, NX-1), same as the single-thread version.
        size_t begin, end;
        thread_range(NX - 2, tid, nt, begin, end);
        stencil_range(a.data(), b.data(), NX, NY, NZ, 0.5f, 1.0f / 12.0f,
                      (int)begin + 1, (int)end + 1);
    });
    sink(b[NX / 2 * NY * NZ]);

    size_t interior = (size_t)(NX - 2) * (NY - 2) * (NZ - 2);
    double flops  = 8.0 * interior;
    double bytes  = 32.0 * interior;
    double ai     = flops / bytes;
    double gflops = flops / best_t / 1e9;

    printf("stencil (MT)  %dx%dx%d, %u threads (best of %d)\n", NX, NY, NZ, nthreads, NTIMES);
    printf("  Time:        %.3f ms\n",        best_t * 1000.0);
    printf("  Arith. Int.: %.4f FLOP/byte\n", ai);
    printf("  Performance: %.4f GFLOP/s\n",  gflops);
    printf("STENCIL_AI_MT=%.4f\n",     ai);
    printf("STENCIL_GFLOPS_MT=%.4f\n", gflops);
    printf("MT_THREADS=%u\n", nthreads);
    return 0;
}
