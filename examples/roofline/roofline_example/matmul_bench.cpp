// Measures FP32 SGEMM performance via either Apple Accelerate or OpenBLAS.
//
// Build (Accelerate):
//   clang++ -std=c++17 -O3 -DACCELERATE_NEW_LAPACK -framework Accelerate -o matmul_bench matmul_bench.cpp
// Build (OpenBLAS):
//   clang++ -std=c++17 -O3 -DUSE_OPENBLAS -I/opt/homebrew/opt/openblas/include \
//           -L/opt/homebrew/opt/openblas/lib -lopenblas -o matmul_bench_ob matmul_bench.cpp
//
// Arithmetic intensity = 2n³ / (3 × n² × sizeof(float)) = n / 6
//   - 2n³ FLOPs: n³ multiplies + n³ adds
//   - 3n² floats: read A, read B, write C (each touched once from memory)

#ifdef USE_OPENBLAS
#  include <cblas.h>
#else
#  include <Accelerate/Accelerate.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace std::chrono;

static const int NTIMES = 5;

int main(int argc, char* argv[]) {
    int n = argc > 1 ? std::atoi(argv[1]) : 1024;

    std::vector<float> A((size_t)n * n), B((size_t)n * n), C((size_t)n * n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    // Warmup: amortise any first-call library startup cost
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0f, A.data(), n, B.data(), n, 0.0f, C.data(), n);

    double best_t = 1e9;
    for (int k = 0; k < NTIMES; k++) {
        auto t0 = high_resolution_clock::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, 1.0f, A.data(), n, B.data(), n, 0.0f, C.data(), n);
        auto t1 = high_resolution_clock::now();
        double t = duration<double>(t1 - t0).count();
        if (t < best_t) best_t = t;
    }

    double flops  = 2.0 * n * n * n;
    double bytes  = 3.0 * n * n * sizeof(float);
    double ai     = flops / bytes;
    double gflops = flops / best_t / 1e9;

#ifdef USE_OPENBLAS
    const char* lib = "OpenBLAS";
#else
    const char* lib = "Accelerate";
#endif

    printf("sgemm %dx%d  [%s]  (best of %d):\n", n, n, lib, NTIMES);
    printf("  Time:        %.3f ms\n",        best_t * 1000.0);
    printf("  FLOP:        %.3f GFLOP\n",     flops / 1e9);
    printf("  Arith. Int.: %.2f FLOP/byte\n", ai);
    printf("  Performance: %.2f GFLOP/s\n",  gflops);
    printf("SGEMM_AI=%.4f\n",     ai);
    printf("SGEMM_GFLOPS=%.4f\n", gflops);
    return 0;
}
