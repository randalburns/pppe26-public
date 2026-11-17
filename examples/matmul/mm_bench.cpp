// mm_bench.cpp -- the kernels from Si Boehm's "Fast Multidimensional Matrix
// Multiplication on CPU from Scratch", one core, FP32, row-major.
//
// The Makefile builds this file three times, because the blog's first step
// is a change of compiler flags, not of code:
//   mm_O0    no flags (clang's default is -O0)
//   mm_O3    -O3 -march=native
//   mm_fast  -O3 -march=native -ffast-math
//
//   ./mm_fast <n> <kernel>...     n is 1024 or 2048
//
// Kernels, in the blog's order (blog name in parentheses):
//   ijk          naive: inner loop walks down a column of B      (RCI)
//   ijk_local    same, accumulating in a local, one store per C   (RCI + register accumulate)
//   ikj          interchange: inner loop walks rows of B and C   (RIC)
//   tile_k<T>    ikj with k blocked by T, kk outermost            (RIC + tiling on I)
//   tile_ijk<T>  i, j and k all blocked by T                      (tiling on multiple dimensions)
//
// Sizes are compile-time constants, as in the blog: the compiler knows every
// trip count. Pointers are not __restrict__, also as in the blog, so the
// aliasing cost that ijk_local removes is still there to remove.
//
// Each kernel prints one line:  RESULT,<build>,<n>,<kernel>,<ms>,<GFLOP/s>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifndef BUILD
#define BUILD "unknown"
#endif

template <int N>
static void mm_ijk(const float* A, const float* B, float* C) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

template <int N>
static void mm_ijk_local(const float* A, const float* B, float* C) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < N; k++) acc += A[i * N + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

template <int N>
static void mm_ikj(const float* A, const float* B, float* C) {
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float a = A[i * N + k];
            for (int j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
        }
}

// The T rows of B in block kk are reused by every row of A before the loop
// moves on to the next block.
template <int N, int T>
static void mm_tile_k(const float* A, const float* B, float* C) {
    for (int kk = 0; kk < N; kk += T)
        for (int i = 0; i < N; i++)
            for (int k = kk; k < kk + T; k++) {
                float a = A[i * N + k];
                for (int j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
            }
}

// kk is inside jj, so a T x T tile of C stays put while the k blocks
// stream through it.
template <int N, int T>
static void mm_tile_ijk(const float* A, const float* B, float* C) {
    for (int ii = 0; ii < N; ii += T)
        for (int jj = 0; jj < N; jj += T)
            for (int kk = 0; kk < N; kk += T)
                for (int i = ii; i < ii + T; i++)
                    for (int k = kk; k < kk + T; k++) {
                        float a = A[i * N + k];
                        for (int j = jj; j < jj + T; j++) C[i * N + j] += a * B[k * N + j];
                    }
}

using Kernel = void (*)(const float*, const float*, float*);

template <int N>
static Kernel lookup(const std::string& name) {
    if (name == "ijk")       return mm_ijk<N>;
    if (name == "ijk_local") return mm_ijk_local<N>;
    if (name == "ikj")       return mm_ikj<N>;
#define TILE(T)                                                     \
    if (name == "tile_k" #T)   return mm_tile_k<N, T>;              \
    if (name == "tile_ijk" #T) return mm_tile_ijk<N, T>;
    TILE(4) TILE(8) TILE(16) TILE(32) TILE(64) TILE(128) TILE(256)
#undef TILE
    return nullptr;
}

// Best of up to 5 runs, stopping early once 3 s have been spent, so the
// -O0 naive kernel at n = 2048 runs once instead of five times.
template <int N>
static int run(int argc, char** argv) {
    std::vector<float> A((size_t)N * N), B((size_t)N * N), C((size_t)N * N), ref((size_t)N * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    // Reference in double, for the correctness check.
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            double a = A[(size_t)i * N + k];
            for (int j = 0; j < N; j++) ref[(size_t)i * N + j] += (float)(a * B[(size_t)k * N + j]);
        }

    const double flop = 2.0 * N * N * N;
    for (int a = 2; a < argc; a++) {
        Kernel fn = lookup<N>(argv[a]);
        if (!fn) { fprintf(stderr, "unknown kernel %s\n", argv[a]); return 1; }

        double best = 1e30, spent = 0;
        for (int rep = 0; rep < 5 && spent < 3.0; rep++) {
            std::fill(C.begin(), C.end(), 0.0f);
            asm volatile("" : : "r"(C.data()) : "memory");
            auto t0 = std::chrono::steady_clock::now();
            fn(A.data(), B.data(), C.data());
            asm volatile("" : : "r"(C.data()) : "memory");
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            best = std::min(best, dt);
            spent += dt;
        }

        double err = 0;
        for (size_t i = 0; i < C.size(); i++) err = std::max(err, (double)std::fabs(C[i] - ref[i]));
        if (err > 1e-2) fprintf(stderr, "WARNING: %s n=%d max error %g\n", argv[a], N, err);

        printf("RESULT,%s,%d,%s,%.2f,%.3f\n", BUILD, N, argv[a], best * 1e3, flop / best / 1e9);
        fflush(stdout);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <1024|2048> <kernel>...\n", argv[0]); return 1; }
    int n = atoi(argv[1]);
    if (n == 1024) return run<1024>(argc, argv);
    if (n == 2048) return run<2048>(argc, argv);
    fprintf(stderr, "n must be 1024 or 2048\n");
    return 1;
}
