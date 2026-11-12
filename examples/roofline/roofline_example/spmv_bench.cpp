// Build: clang++ -std=c++17 -O3 -march=native -o spmv_bench spmv_bench.cpp
// Measures FP32 sparse matrix-vector product (CSR, y = A*x) — the classic
// sparse-linear-algebra kernel from the original roofline paper.
//
// Column indices are uniform-random across all columns (not banded), so the
// gather into x[] is irregular — deliberately the "worst case" SpMV access
// pattern. A banded/structured sparsity pattern would show much better
// locality and land closer to the bandwidth roofline than this does.
//
// Arithmetic intensity = 2 FLOP (mul+add) per nonzero /
//                         12 bytes per nonzero (4B val + 4B col_idx + 4B gathered x)
//                      = 0.1667 FLOP/byte
//   (row_ptr/y traffic is O(nrows), negligible next to O(nnz) here since
//   nnz_per_row is small)

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace std::chrono;

template <typename T>
static void sink(T const& v) { asm volatile("" : : "m"(v) : "memory"); }

static const size_t NROWS       = 4ULL * 1024 * 1024; // 4M rows -> x = 16 MB (> SLC)
static const int    NNZ_PER_ROW = 7;
static const int    NTIMES      = 5;

__attribute__((noinline))
static void spmv(const int64_t* __restrict__ row_ptr, const int32_t* __restrict__ col_idx,
                  const float* __restrict__ val, const float* __restrict__ x,
                  float* __restrict__ y, size_t nrows) {
    asm volatile("" : : : "memory");
    for (size_t i = 0; i < nrows; i++) {
        float sum = 0.0f;
        for (int64_t k = row_ptr[i]; k < row_ptr[i + 1]; k++) sum += val[k] * x[col_idx[k]];
        y[i] = sum;
    }
}

int main() {
    size_t nnz = NROWS * NNZ_PER_ROW;
    std::vector<int64_t> row_ptr(NROWS + 1);
    std::vector<int32_t> col_idx(nnz);
    std::vector<float>   val(nnz), x(NROWS), y(NROWS);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float>  vdist(-1.0f, 1.0f);
    std::uniform_int_distribution<int32_t> cdist(0, (int32_t)NROWS - 1);

    for (size_t i = 0; i <= NROWS; i++) row_ptr[i] = (int64_t)i * NNZ_PER_ROW;
    for (size_t k = 0; k < nnz; k++) { val[k] = vdist(rng); col_idx[k] = cdist(rng); }
    for (auto& v : x) v = vdist(rng);

    double best_t = 1e9;
    for (int k = 0; k < NTIMES; k++) {
        auto t0 = high_resolution_clock::now();
        spmv(row_ptr.data(), col_idx.data(), val.data(), x.data(), y.data(), NROWS);
        auto t1 = high_resolution_clock::now();
        double t = duration<double>(t1 - t0).count();
        if (t < best_t) best_t = t;
    }
    sink(y[0]);

    double flops  = 2.0 * nnz;
    double bytes  = 12.0 * nnz;
    double ai     = flops / bytes;
    double gflops = flops / best_t / 1e9;

    printf("spmv  %zuM rows x %d nnz/row (best of %d)\n", NROWS >> 20, NNZ_PER_ROW, NTIMES);
    printf("  Time:        %.3f ms\n",        best_t * 1000.0);
    printf("  Arith. Int.: %.4f FLOP/byte\n", ai);
    printf("  Performance: %.4f GFLOP/s\n",  gflops);
    printf("SPMV_AI=%.4f\n",     ai);
    printf("SPMV_GFLOPS=%.4f\n", gflops);
    return 0;
}
