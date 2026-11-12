// Build: clang++ -std=c++17 -O3 -march=native -o stencil_bench stencil_bench.cpp
// Measures FP32 7-point stencil (3D Jacobi update) performance — the classic
// structured-grid kernel from the original roofline paper.
//
// out[c] = c0*in[c] + c1*(in[c-1]+in[c+1]+in[c-ny]+in[c+ny]+in[c-nynz]+in[c+nynz])
//
// Arithmetic intensity = 8 FLOP (5 add to sum 6 neighbors + 2 mul + 1 add to
//                         combine) / 32 bytes (7 reads + 1 write, compulsory-
//                         touch model) = 0.25 FLOP/byte
//   Real DRAM traffic is lower in practice: consecutive k reuse cache lines
//   for the +-1 neighbors, so this AI is a lower bound on achievable
//   throughput, not an exact count — same caveat as sort_bench's.
//
// Grid is sized well beyond the M5 SLC (12 MB per buffer) so reads come from DRAM.

#include <chrono>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

using namespace std::chrono;

template <typename T>
static void sink(T const& v) { asm volatile("" : : "m"(v) : "memory"); }

static const int NX = 256, NY = 256, NZ = 256; // 256^3 * 4B = 64 MB per buffer
static const int NTIMES = 5;

__attribute__((noinline))
static void stencil(const float* __restrict__ in, float* __restrict__ out,
                     int nx, int ny, int nz, float c0, float c1) {
    asm volatile("" : : : "memory");
    size_t nynz = (size_t)ny * nz;
    for (int i = 1; i < nx - 1; i++) {
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
    size_t n = (size_t)NX * NY * NZ;
    std::vector<float> a(n), b(n, 0.0f);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : a) v = dist(rng);

    double best_t = 1e9;
    for (int k = 0; k < NTIMES; k++) {
        auto t0 = high_resolution_clock::now();
        stencil(a.data(), b.data(), NX, NY, NZ, 0.5f, 1.0f / 12.0f);
        auto t1 = high_resolution_clock::now();
        double t = duration<double>(t1 - t0).count();
        if (t < best_t) best_t = t;
        std::swap(a, b); // ping-pong: next call sees fresh input, not a repeat
    }
    sink(a[0]);

    size_t interior = (size_t)(NX - 2) * (NY - 2) * (NZ - 2);
    double flops  = 8.0 * interior;
    double bytes  = 32.0 * interior;
    double ai     = flops / bytes;
    double gflops = flops / best_t / 1e9;

    printf("stencil  %dx%dx%d (best of %d)\n", NX, NY, NZ, NTIMES);
    printf("  Time:        %.3f ms\n",        best_t * 1000.0);
    printf("  Arith. Int.: %.4f FLOP/byte\n", ai);
    printf("  Performance: %.4f GFLOP/s\n",  gflops);
    printf("STENCIL_AI=%.4f\n",     ai);
    printf("STENCIL_GFLOPS=%.4f\n", gflops);
    return 0;
}
