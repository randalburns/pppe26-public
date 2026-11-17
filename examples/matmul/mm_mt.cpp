// mm_mt.cpp -- multithreaded matrix multiply, FP32, row-major: partition the
// output C, or partition the inputs A and B along k and merge the partial
// products.
//
//   ./mm_mt <n> <threads> <variant>...     n is 1024 or 2048
//
// Variants:
//   c_rows        thread t owns a band of n/p rows of C. Every thread reads
//                 all of B; no thread writes another's rows. No
//                 synchronization.
//   c_rows_tiled  c_rows with the single-core tile_k inner loop (k blocked
//                 by 16) instead of partial_row: the same partition, the
//                 best single-core kernel inside it.
//   c_cols_tiled  thread t owns a band of n/p columns of C: it reads all of
//                 A and only its column band of B.
//   c_square_tiled  thread t owns one block of a pr x pc grid over C, as
//                 square as p allows (2 x 2 at p = 4; at p = 2 it is two
//                 column halves): it reads half of A and half of B.
//   c_tiles256    the blog's decomposition: 256 x 256 tiles of C, dealt to
//                 the threads round-robin.
//                 All four *_tiled shapes use the same kernel, region_tiled.
//   k_atomic      thread t owns a band of n/p values of k: columns of A, rows
//                 of B. Its partial product covers all of C, and it adds
//                 every term into C with an atomic add: n^3 atomics.
//   k_atomic_row  as k_atomic, but each row of the partial product is summed
//                 in a private buffer first and added into C once: n^2
//                 atomics per thread.
//   k_atomic_stagger  as k_atomic_row, but thread t starts on row t*n/p, so
//                 the threads add into different rows at any moment.
//   k_private     as k_atomic_row, but each thread writes its whole partial
//                 product to a private copy of C; a second parallel pass sums
//                 the p copies into C. No atomics; p copies of C.
//
// The split is static and even, so run with at most as many threads as
// performance cores (4 on the M5): on efficiency cores the slowest thread
// sets the time, and that is a placement question, not a partitioning one.
//
// Every variant uses the same inner loop, partial_row (ikj over a range of k,
// one row of C at a time), so with one thread they all run the same code and
// the differences at p threads come from the partitioning alone.
//
// Each run prints:
//   RESULT_MT,<n>,<variant>,<p>,<ms>,<GFLOP/s>,<multiply ms>,<merge ms>
// where the two phase times are filled in for k_private only.

#include "../roofline/roofline_example/mt_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// AArch64 has atomic integer add (LDADD) but no atomic floating-point add,
// so a float atomic add is a compare-and-swap loop on the bits (Lecture 14).
static inline void atomic_add(float* p, float v) {
    uint32_t* q = reinterpret_cast<uint32_t*>(p);
    uint32_t old = __atomic_load_n(q, __ATOMIC_RELAXED);
    for (;;) {
        float f;
        std::memcpy(&f, &old, 4);
        f += v;
        uint32_t upd;
        std::memcpy(&upd, &f, 4);
        if (__atomic_compare_exchange_n(q, &old, upd, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return;
    }
}

template <int N>
static void k_atomic(const float* A, const float* B, float* C, int k0, int k1) {
    for (int i = 0; i < N; i++)
        for (int k = k0; k < k1; k++) {
            float a = A[i * N + k];
            for (int j = 0; j < N; j++) atomic_add(&C[i * N + j], a * B[k * N + j]);
        }
}

// Row i of A[:, k0:k1] x B[k0:k1, :], summed in acc (one row, in L1).
template <int N>
static void partial_row(const float* A, const float* B, float* acc, int i, int k0, int k1) {
    std::fill(acc, acc + N, 0.0f);
    for (int k = k0; k < k1; k++) {
        float a = A[i * N + k];
        for (int j = 0; j < N; j++) acc[j] += a * B[k * N + j];
    }
}

// Rows [r0, r1) of C, each over the full range of k, written in place.
template <int N>
static void c_rows(const float* A, const float* B, float* C, int r0, int r1) {
    for (int i = r0; i < r1; i++) partial_row<N>(A, B, C + (size_t)i * N, i, 0, N);
}

// Rows [r0, r1) x columns [c0, c1) of C, with k blocked by 16: the
// single-core tile_k kernel restricted to one rectangle.
template <int N>
static void region_tiled(const float* A, const float* B, float* C, int r0, int r1, int c0, int c1) {
    const int T = 16;
    for (int kk = 0; kk < N; kk += T)
        for (int i = r0; i < r1; i++)
            for (int k = kk; k < kk + T; k++) {
                float a = A[i * N + k];
                for (int j = c0; j < c1; j++) C[i * N + j] += a * B[k * N + j];
            }
}

// first: the row this thread starts on. k_atomic_row starts every thread on
// row 0, so all p threads add into the same row of C at the same time;
// k_atomic_stagger starts thread t on row t*n/p, so at any moment each
// thread is adding into a different row.
template <int N>
static void k_atomic_row(const float* A, const float* B, float* C, int k0, int k1, int first) {
    std::vector<float> acc(N);
    for (int r = 0; r < N; r++) {
        int i = (first + r) % N;
        partial_row<N>(A, B, acc.data(), i, k0, k1);
        for (int j = 0; j < N; j++) atomic_add(&C[i * N + j], acc[j]);
    }
}

template <int N>
static void k_private_multiply(const float* A, const float* B, float* Cp, int k0, int k1) {
    for (int i = 0; i < N; i++) partial_row<N>(A, B, Cp + (size_t)i * N, i, k0, k1);
}

static void split(int n, unsigned t, unsigned p, int& lo, int& hi) {
    size_t b, e;
    thread_range((size_t)n, t, p, b, e);
    lo = (int)b;
    hi = (int)e;
}

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <int N>
static int run(unsigned p, int argc, char** argv) {
    const size_t NN = (size_t)N * N;
    std::vector<float> A(NN), B(NN), C(NN), ref(NN);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            double a = A[(size_t)i * N + k];
            for (int j = 0; j < N; j++) ref[(size_t)i * N + j] += (float)(a * B[(size_t)k * N + j]);
        }
    std::vector<float> priv;  // p private copies of C, for k_private

    const double flop = 2.0 * N * N * N;
    for (int a = 3; a < argc; a++) {
        std::string v = argv[a];
        if (v == "k_private") priv.assign(NN * p, 0.0f);

        double best = 1e30, best_mul = 0, best_merge = 0, spent = 0;
        for (int rep = 0; rep < 5 && spent < 3.0; rep++) {
            std::fill(C.begin(), C.end(), 0.0f);
            double t0 = now(), t1 = t0;
            if (v == "c_rows") {
                run_on_threads(p, [&](unsigned t, unsigned np) {
                    int lo, hi; split(N, t, np, lo, hi);
                    c_rows<N>(A.data(), B.data(), C.data(), lo, hi);
                });
            } else if (v == "c_rows_tiled" || v == "c_cols_tiled") {
                bool cols = v == "c_cols_tiled";
                run_on_threads(p, [&](unsigned t, unsigned np) {
                    int lo, hi; split(N, t, np, lo, hi);
                    if (cols) region_tiled<N>(A.data(), B.data(), C.data(), 0, N, lo, hi);
                    else      region_tiled<N>(A.data(), B.data(), C.data(), lo, hi, 0, N);
                });
            } else if (v == "c_square_tiled") {
                unsigned pr = 1;  // largest divisor of p no greater than sqrt(p)
                for (unsigned d = 1; d * d <= p; d++) if (p % d == 0) pr = d;
                unsigned pc = p / pr;
                run_on_threads(p, [&](unsigned t, unsigned) {
                    int r0, r1, c0, c1;
                    split(N, t / pc, pr, r0, r1);
                    split(N, t % pc, pc, c0, c1);
                    region_tiled<N>(A.data(), B.data(), C.data(), r0, r1, c0, c1);
                });
            } else if (v == "c_tiles256") {
                const int S = 256, per = N / S;
                run_on_threads(p, [&](unsigned t, unsigned np) {
                    for (int tile = (int)t; tile < per * per; tile += (int)np) {
                        int r0 = tile / per * S, c0 = tile % per * S;
                        region_tiled<N>(A.data(), B.data(), C.data(), r0, r0 + S, c0, c0 + S);
                    }
                });
            } else if (v == "k_atomic") {
                run_on_threads(p, [&](unsigned t, unsigned np) {
                    int lo, hi; split(N, t, np, lo, hi);
                    k_atomic<N>(A.data(), B.data(), C.data(), lo, hi);
                });
            } else if (v == "k_atomic_row" || v == "k_atomic_stagger") {
                bool stagger = v == "k_atomic_stagger";
                run_on_threads(p, [&](unsigned t, unsigned np) {
                    int lo, hi; split(N, t, np, lo, hi);
                    int first = stagger ? (int)((size_t)t * N / np) : 0;
                    k_atomic_row<N>(A.data(), B.data(), C.data(), lo, hi, first);
                });
            } else if (v == "k_private") {
                run_on_threads(p, [&](unsigned t, unsigned np) {
                    int lo, hi; split(N, t, np, lo, hi);
                    k_private_multiply<N>(A.data(), B.data(), priv.data() + NN * t, lo, hi);
                });
                t1 = now();
                // Merge: thread t sums the p copies over its band of rows.
                run_on_threads(p, [&](unsigned t, unsigned np) {
                    int lo, hi; split(N, t, np, lo, hi);
                    for (unsigned s = 0; s < np; s++) {
                        const float* src = priv.data() + NN * s;
                        for (size_t x = (size_t)lo * N; x < (size_t)hi * N; x++) C[x] += src[x];
                    }
                });
            } else {
                fprintf(stderr, "unknown variant %s\n", v.c_str());
                return 1;
            }
            double t2 = now(), dt = t2 - t0;
            if (dt < best) { best = dt; best_mul = t1 - t0; best_merge = t2 - t1; }
            spent += dt;
        }

        double err = 0;
        for (size_t x = 0; x < NN; x++) err = std::max(err, (double)std::fabs(C[x] - ref[x]));
        if (err > 1e-2) fprintf(stderr, "WARNING: %s n=%d p=%u max error %g\n", v.c_str(), N, p, err);

        bool phases = v == "k_private";
        printf("RESULT_MT,%d,%s,%u,%.2f,%.3f,%.2f,%.2f\n", N, v.c_str(), p, best * 1e3, flop / best / 1e9,
               phases ? best_mul * 1e3 : 0.0, phases ? best_merge * 1e3 : 0.0);
        fflush(stdout);
        priv.clear();
        priv.shrink_to_fit();
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <1024|2048> <threads> <variant>...\n", argv[0]); return 1; }
    int n = atoi(argv[1]);
    unsigned p = (unsigned)atoi(argv[2]);
    if (n == 1024) return run<1024>(p, argc, argv);
    if (n == 2048) return run<2048>(p, argc, argv);
    fprintf(stderr, "n must be 1024 or 2048\n");
    return 1;
}
