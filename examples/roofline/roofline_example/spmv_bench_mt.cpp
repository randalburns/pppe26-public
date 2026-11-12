// Build: clang++ -std=c++17 -O3 -march=native -o spmv_bench_mt spmv_bench_mt.cpp
// Multicore companion to spmv_bench.cpp: partitions matrix rows across
// threads (each row's dot product is independent, so this is safe without
// synchronization — every thread writes disjoint y[i]).
//
// spmv_bench.cpp measured well below even its own (already low) bandwidth-
// bound line, because a single core's memory-level parallelism can't keep
// enough of the irregular x[col_idx[k]] gathers in flight to hide their
// latency. Multiple cores each issuing independent gathers concurrently
// might recover some of that gap even though the kernel is memory-bound —
// unlike dot_bench_mt.cpp/saxpy_bench_mt.cpp's sequential streams, which
// already saturated bandwidth on a single core and had nothing to gain.

#include "mt_common.hpp"
#include <cstdint>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

template <typename T>
static void sink(T const& v) { asm volatile("" : : "m"(v) : "memory"); }

static const size_t NROWS       = 4ULL * 1024 * 1024; // same as spmv_bench.cpp
static const int    NNZ_PER_ROW = 7;
static const int    NTIMES      = 5;

__attribute__((noinline))
static void spmv_range(const int64_t* __restrict__ row_ptr, const int32_t* __restrict__ col_idx,
                        const float* __restrict__ val, const float* __restrict__ x,
                        float* __restrict__ y, size_t row_begin, size_t row_end) {
    asm volatile("" : : : "memory");
    for (size_t i = row_begin; i < row_end; i++) {
        float sum = 0.0f;
        for (int64_t k = row_ptr[i]; k < row_ptr[i + 1]; k++) sum += val[k] * x[col_idx[k]];
        y[i] = sum;
    }
}

int main() {
    unsigned nthreads = std::thread::hardware_concurrency();
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

    double best_t = best_of_mt(nthreads, NTIMES, [&](unsigned tid, unsigned nt) {
        size_t begin, end;
        thread_range(NROWS, tid, nt, begin, end);
        spmv_range(row_ptr.data(), col_idx.data(), val.data(), x.data(), y.data(), begin, end);
    });
    sink(y[0]);

    double flops  = 2.0 * nnz;
    double bytes  = 12.0 * nnz;
    double ai     = flops / bytes;
    double gflops = flops / best_t / 1e9;

    printf("spmv (MT)  %zuM rows x %d nnz/row, %u threads (best of %d)\n",
           NROWS >> 20, NNZ_PER_ROW, nthreads, NTIMES);
    printf("  Time:        %.3f ms\n",        best_t * 1000.0);
    printf("  Arith. Int.: %.4f FLOP/byte\n", ai);
    printf("  Performance: %.4f GFLOP/s\n",  gflops);
    printf("SPMV_AI_MT=%.4f\n",     ai);
    printf("SPMV_GFLOPS_MT=%.4f\n", gflops);
    printf("MT_THREADS=%u\n", nthreads);
    return 0;
}
