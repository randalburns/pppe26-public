// Build: clang++ -std=c++17 -O3 -march=native -o saxpy_bench saxpy_bench.cpp
// Measures FP32 SAXPY (y = a*x + y) performance — the classic BLAS-1 kernel.
//
// Arithmetic intensity = 2 FLOP (mul+add) / 12 bytes (read x[i], read+write y[i])
//                      = 0.1667 FLOP/byte  ->  memory-bound, close to STREAM Scale/Triad
//
// Arrays are sized well beyond the M5 SLC (12 MB) so reads come from DRAM.

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace std::chrono;

template <typename T>
static void sink(T const& v) { asm volatile("" : : "m"(v) : "memory"); }

static const size_t N      = 32ULL * 1024 * 1024; // 32M floats = 128 MB per array
static const int    NTIMES = 5;

__attribute__((noinline))
static void saxpy(float a, const float* __restrict__ x, float* __restrict__ y, size_t n) {
    asm volatile("" : : : "memory");
    for (size_t i = 0; i < n; i++) y[i] = a * x[i] + y[i];
}

int main() {
    std::vector<float> x(N), y(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x) v = dist(rng);
    for (auto& v : y) v = dist(rng);

    double best_t = 1e9;
    for (int k = 0; k < NTIMES; k++) {
        auto t0 = high_resolution_clock::now();
        saxpy(2.0f, x.data(), y.data(), N);
        auto t1 = high_resolution_clock::now();
        double t = duration<double>(t1 - t0).count();
        if (t < best_t) best_t = t;
    }
    sink(y[0]);

    double flops  = 2.0 * N;
    double bytes  = 3.0 * N * sizeof(float); // read x, read y, write y
    double ai     = flops / bytes;
    double gflops = flops / best_t / 1e9;

    printf("saxpy  N=%zuM floats (best of %d)\n", N >> 20, NTIMES);
    printf("  Time:        %.3f ms\n",        best_t * 1000.0);
    printf("  Arith. Int.: %.4f FLOP/byte\n", ai);
    printf("  Performance: %.4f GFLOP/s\n",  gflops);
    printf("SAXPY_AI=%.4f\n",     ai);
    printf("SAXPY_GFLOPS=%.4f\n", gflops);
    return 0;
}
