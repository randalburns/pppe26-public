// mm_sme.cpp -- matrix multiply on the Apple matrix unit, with Arm SME.
//
// An add-on to ../mm_bench.cpp. Apple silicon M4 or later only.
//
//   ./mm_sme <n> <kernel>...      n a multiple of 32, e.g. 1024 or 2048
//   ./mm_sme peak                 the unit's peak rate, with 1, 2 and 4 tiles
//
// Kernels:
//   sme1            one 16 x 16 tile of C. A is transposed, all of it, first
//   sme4            four tiles, a 32 x 32 block of C. Same transpose
//   sme4_pack_k<K>  four tiles; A and B are packed a slab of K values of k
//                   at a time, inside the matrix unit, so the kernel reads
//                   small buffers in order. K a multiple of 16 that divides n
//   sme4_packA_k<K>     four tiles; all of A is packed once, up front, and B
//                       is read where it is. k still runs in slabs of K
//   sme4_prepacked_k<K> the same multiply, with A packed before the clock
//                       starts: what a second multiply by the same A costs
//   accelerate      Apple's cblas_sgemm, for comparison
//   transpose       only the transpose that sme1 and sme4 do (not a multiply)
//
// SME in one paragraph. FMOPA takes two vectors of 16 floats, a and b, and
// adds a[r] * b[c] into element (r, c) of a 16 x 16 tile: 256 multiply-adds.
// The tiles live in ZA, storage inside the matrix unit; with 32-bit floats
// there are four of them. A function has to be in "streaming mode" to use
// them: __arm_locally_streaming switches on entry and back on return, and
// __arm_new("za") gives the function a fresh ZA.
//
// For C = A x B, one step of k adds (column k of A) x (row k of B) to C.
// That is one FMOPA per 16 x 16 block of C. A row of B is contiguous in
// memory. A column of A is not, and streaming mode has no gather load, so
// every kernel here first rearranges A so that its columns are contiguous.
//
// Each kernel prints:  RESULT,sme,<n>,<kernel>,<ms>,<GFLOP/s>

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <arm_sme.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Buffers the kernels reuse from one call to the next, as a library would.
static std::vector<float> g_At, g_Ap, g_Bp;

// At = A transposed: At[k][i] = A[i][k]. Done in 32 x 32 tiles so that
// both the reads and the writes stay within a few cache lines (Lecture 6).
static void transpose(const float* A, float* At, int n) {
    for (int ii = 0; ii < n; ii += 32)
        for (int kk = 0; kk < n; kk += 32)
            for (int i = ii; i < ii + 32; i++)
                for (int k = kk; k < kk + 32; k++) At[(size_t)k * n + i] = A[(size_t)i * n + k];
}

// ---- sme1: one tile ---------------------------------------------------
//
// The 16 x 16 block of C at (i, j) is tile 0. Load it, add one outer
// product for every k, store it. Two vector loads per FMOPA.
__arm_locally_streaming __arm_new("za")
static void kernel1(const float* At, const float* B, float* C, int n) {
    const int vl = (int)svcntw();  // floats per vector: 16 on the M4 and M5
    const svbool_t pg = svptrue_b32();
    for (int i = 0; i < n; i += vl)
        for (int j = 0; j < n; j += vl) {
            for (int r = 0; r < vl; r++) svld1_hor_za32(0, r, pg, &C[(size_t)(i + r) * n + j]);
            for (int k = 0; k < n; k++) {
                svfloat32_t a = svld1_f32(pg, &At[(size_t)k * n + i]);  // A[i..i+15][k]
                svfloat32_t b = svld1_f32(pg, &B[(size_t)k * n + j]);   // B[k][j..j+15]
                svmopa_za32_f32_m(0, pg, pg, a, b);
            }
            for (int r = 0; r < vl; r++) svst1_hor_za32(0, r, pg, &C[(size_t)(i + r) * n + j]);
        }
}

static void mm_sme1(const float* A, const float* B, float* C, int n) {
    g_At.resize((size_t)n * n);
    transpose(A, g_At.data(), n);
    kernel1(g_At.data(), B, C, n);
}

// ---- sme4: four tiles ---------------------------------------------------
//
// The 32 x 32 block of C at (i, j) is all four tiles:
//
//              b0 (cols j..j+15)   b1 (cols j+16..j+31)
//   a0 (rows i..i+15)      tile 0            tile 1
//   a1 (rows i+16..i+31)   tile 2            tile 3
//
// Each k step loads two vectors of A and two of B, and each is used by two
// FMOPAs: four loads feed 1,024 multiply-adds. The four tiles are also
// four independent accumulators, so one FMOPA need not wait for the last.
#define LOAD_C4(Cp, n, vl, pg)                                                         \
    for (int r = 0; r < vl; r++) {                                                     \
        svld1_hor_za32(0, r, pg, (Cp) + (size_t)r * (n));                              \
        svld1_hor_za32(1, r, pg, (Cp) + (size_t)r * (n) + vl);                         \
        svld1_hor_za32(2, r, pg, (Cp) + (size_t)(r + vl) * (n));                       \
        svld1_hor_za32(3, r, pg, (Cp) + (size_t)(r + vl) * (n) + vl);                  \
    }
#define STORE_C4(Cp, n, vl, pg)                                                        \
    for (int r = 0; r < vl; r++) {                                                     \
        svst1_hor_za32(0, r, pg, (Cp) + (size_t)r * (n));                              \
        svst1_hor_za32(1, r, pg, (Cp) + (size_t)r * (n) + vl);                         \
        svst1_hor_za32(2, r, pg, (Cp) + (size_t)(r + vl) * (n));                       \
        svst1_hor_za32(3, r, pg, (Cp) + (size_t)(r + vl) * (n) + vl);                  \
    }
#define FMOPA4(a0, a1, b0, b1, pg)                                                     \
    svmopa_za32_f32_m(0, pg, pg, a0, b0);                                              \
    svmopa_za32_f32_m(1, pg, pg, a0, b1);                                              \
    svmopa_za32_f32_m(2, pg, pg, a1, b0);                                              \
    svmopa_za32_f32_m(3, pg, pg, a1, b1);

__arm_locally_streaming __arm_new("za")
static void kernel4(const float* At, const float* B, float* C, int n) {
    const int vl = (int)svcntw();
    const svbool_t pg = svptrue_b32();
    for (int i = 0; i < n; i += 2 * vl)
        for (int j = 0; j < n; j += 2 * vl) {
            float* Cp = &C[(size_t)i * n + j];
            LOAD_C4(Cp, n, vl, pg)
            for (int k = 0; k < n; k++) {
                svfloat32_t a0 = svld1_f32(pg, &At[(size_t)k * n + i]);
                svfloat32_t a1 = svld1_f32(pg, &At[(size_t)k * n + i + vl]);
                svfloat32_t b0 = svld1_f32(pg, &B[(size_t)k * n + j]);
                svfloat32_t b1 = svld1_f32(pg, &B[(size_t)k * n + j + vl]);
                FMOPA4(a0, a1, b0, b1, pg)
            }
            STORE_C4(Cp, n, vl, pg)
        }
}

static void mm_sme4(const float* A, const float* B, float* C, int n) {
    g_At.resize((size_t)n * n);
    transpose(A, g_At.data(), n);
    kernel4(g_At.data(), B, C, n);
}

// ---- sme4_pack: four tiles, packed inputs, k in slabs -------------------
//
// The whole-matrix transpose is the problem with sme4: it costs a quarter of
// the run at n = 1024, and more than the multiply itself at n = 2048. So
// rearrange only the part of A and B about to be used, a slab of K values
// of k at a time, into two buffers laid out in the order the kernel reads:
//
//   Ap[ib][k][32]   the 32 rows of A in block ib, transposed: 32 floats per k
//   Bp[jb][k][32]   the 32 columns of B in block jb: 32 floats per k
//
// The kernel then reads two sequential streams for each 32 x 32 block of C.
//
// The packing runs in the matrix unit too. A tile can be loaded by rows and
// stored by columns, so two tiles transpose a 32 x 16 piece of A with 32
// vector loads and 32 vector stores, and no scalar code. B is copied with
// vector loads and stores. (A first version packed with ordinary scalar
// loops outside streaming mode: packing was then 30-40% of the run time,
// and the kernel ran at 1,050-1,130 GFLOP/s instead of 1,700-1,870.)
//
// The first slab starts from zeroed tiles instead of loading C, so this
// kernel computes C = A x B, as sgemm does with beta = 0.
__arm_locally_streaming __arm_new("za")
static void slab(const float* A, const float* B, float* C, int n, int kk, int K, float* Ap,
                 float* Bp) {
    const int vl = (int)svcntw();
    const int W = 2 * vl;  // block width: 32
    const svbool_t pg = svptrue_b32();

    // Pack A: load 16 columns of 32 rows into tiles 0 and 1 by rows, and
    // store them by columns.
    for (int ib = 0; ib < n / W; ib++) {
        float* dst = &Ap[(size_t)ib * K * W];
        for (int k0 = 0; k0 < K; k0 += vl) {
            for (int r = 0; r < vl; r++) {
                svld1_hor_za32(0, r, pg, &A[(size_t)(ib * W + r) * n + kk + k0]);
                svld1_hor_za32(1, r, pg, &A[(size_t)(ib * W + vl + r) * n + kk + k0]);
            }
            for (int c = 0; c < vl; c++) {
                svst1_ver_za32(0, c, pg, dst + (size_t)(k0 + c) * W);
                svst1_ver_za32(1, c, pg, dst + (size_t)(k0 + c) * W + vl);
            }
        }
    }

    // Pack B: rows kk..kk+K, cut into panels 32 columns wide.
    for (int k = 0; k < K; k++) {
        const float* row = &B[(size_t)(kk + k) * n];
        for (int jb = 0; jb < n / W; jb++) {
            float* d = &Bp[((size_t)jb * K + k) * W];
            svst1_f32(pg, d, svld1_f32(pg, row + (size_t)jb * W));
            svst1_f32(pg, d + vl, svld1_f32(pg, row + (size_t)jb * W + vl));
        }
    }

    // Multiply: K outer products into each 32 x 32 block of C.
    for (int ib = 0; ib < n / W; ib++)
        for (int jb = 0; jb < n / W; jb++) {
            float* Cp = &C[(size_t)ib * W * n + (size_t)jb * W];
            const float* a = &Ap[(size_t)ib * K * W];
            const float* b = &Bp[(size_t)jb * K * W];
            if (kk == 0) {
                svzero_za();
            } else {
                LOAD_C4(Cp, n, vl, pg)
            }
            for (int k = 0; k < K; k++) {
                svfloat32_t a0 = svld1_f32(pg, a + (size_t)k * W);
                svfloat32_t a1 = svld1_f32(pg, a + (size_t)k * W + vl);
                svfloat32_t b0 = svld1_f32(pg, b + (size_t)k * W);
                svfloat32_t b1 = svld1_f32(pg, b + (size_t)k * W + vl);
                FMOPA4(a0, a1, b0, b1, pg)
            }
            STORE_C4(Cp, n, vl, pg)
        }
}

static void mm_sme4_pack(const float* A, const float* B, float* C, int n, int K) {
    g_Ap.resize((size_t)n * K);
    g_Bp.resize((size_t)n * K);
    for (int kk = 0; kk < n; kk += K) slab(A, B, C, n, kk, K, g_Ap.data(), g_Bp.data());
}

// ---- sme4_packA: pack A once, read B in place ---------------------------
//
// B's rows are already contiguous, so B needs no rearranging: only A does.
// pack_A transposes all of A, through the tiles as above, into
//
//   Ap[ib][k][32]   for every block ib of 32 rows and every k
//
// It is done once. The packed A can then be kept and used again, for
// another multiply by the same A.
//
// The multiply reads Ap in order. B is read in place: for one block of C
// the kernel walks down a strip of B 32 columns wide, one row (n floats)
// apart per k. k still runs in slabs of K so that the K rows of B in use
// stay in the L2 while every block of C uses them.
__arm_locally_streaming __arm_new("za")
static void pack_A(const float* A, float* Ap, int n) {
    const int vl = (int)svcntw();
    const int W = 2 * vl;
    const svbool_t pg = svptrue_b32();
    for (int ib = 0; ib < n / W; ib++) {
        float* dst = &Ap[(size_t)ib * n * W];
        for (int k0 = 0; k0 < n; k0 += vl) {
            for (int r = 0; r < vl; r++) {
                svld1_hor_za32(0, r, pg, &A[(size_t)(ib * W + r) * n + k0]);
                svld1_hor_za32(1, r, pg, &A[(size_t)(ib * W + vl + r) * n + k0]);
            }
            for (int c = 0; c < vl; c++) {
                svst1_ver_za32(0, c, pg, dst + (size_t)(k0 + c) * W);
                svst1_ver_za32(1, c, pg, dst + (size_t)(k0 + c) * W + vl);
            }
        }
    }
}

__arm_locally_streaming __arm_new("za")
static void multiply_packedA(const float* Ap, const float* B, float* C, int n, int K) {
    const int vl = (int)svcntw();
    const int W = 2 * vl;
    const svbool_t pg = svptrue_b32();
    for (int kk = 0; kk < n; kk += K)
        for (int ib = 0; ib < n / W; ib++)
            for (int jb = 0; jb < n / W; jb++) {
                float* Cp = &C[(size_t)ib * W * n + (size_t)jb * W];
                const float* a = &Ap[((size_t)ib * n + kk) * W];
                const float* b = &B[(size_t)kk * n + (size_t)jb * W];
                if (kk == 0) {
                    svzero_za();
                } else {
                    LOAD_C4(Cp, n, vl, pg)
                }
                for (int k = 0; k < K; k++) {
                    svfloat32_t a0 = svld1_f32(pg, a + (size_t)k * W);
                    svfloat32_t a1 = svld1_f32(pg, a + (size_t)k * W + vl);
                    svfloat32_t b0 = svld1_f32(pg, b + (size_t)k * n);
                    svfloat32_t b1 = svld1_f32(pg, b + (size_t)k * n + vl);
                    FMOPA4(a0, a1, b0, b1, pg)
                }
                STORE_C4(Cp, n, vl, pg)
            }
}

// ---- peak: FMOPAs with no memory at all ---------------------------------
//
// How fast can the unit go? The same two vectors, added into 1, 2 or 4
// tiles over and over. If FMOPAs into one tile must wait for each other,
// more tiles go faster: the accumulator-count argument of Lecture 17.
__arm_locally_streaming __arm_new("za")
static double peak(int tiles, long iters) {
    const svbool_t pg = svptrue_b32();
    svfloat32_t a = svdup_f32(1.0001f), b = svdup_f32(0.5f);
    svzero_za();
    for (long it = 0; it < iters; it++) {
        svmopa_za32_f32_m(0, pg, pg, a, b);
        if (tiles >= 2) svmopa_za32_f32_m(1, pg, pg, a, b);
        if (tiles >= 4) {
            svmopa_za32_f32_m(2, pg, pg, a, b);
            svmopa_za32_f32_m(3, pg, pg, a, b);
        }
    }
    float out[64];
    svst1_hor_za32(0, 0, pg, out);
    svst1_hor_za32(3, 0, pg, out + 16);
    return out[0] + out[16];
}

static void run_peak() {
    const int vl = 16;
    for (int tiles : {1, 2, 4}) {
        long iters = 200'000'000;
        double best = 1e30, sink = 0;
        for (int rep = 0; rep < 3; rep++) {
            double t0 = now();
            sink += peak(tiles, iters);
            best = std::min(best, now() - t0);
        }
        double flop = 2.0 * vl * vl * tiles * iters;  // 256 multiply-adds per FMOPA
        printf("PEAK,sme,%d tiles,%.0f GFLOP/s,%.2f FMOPA/ns%s\n", tiles, flop / best / 1e9,
               tiles * iters / best / 1e9, sink < 0 ? "!" : "");
    }
}

// ---- harness -------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "peak") { run_peak(); return 0; }
    if (argc < 3) {
        fprintf(stderr, "usage: %s <n> <kernel>...   or   %s peak\n", argv[0], argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    if (n < 32 || n % 32) { fprintf(stderr, "n must be a multiple of 32\n"); return 1; }
    if (svcntsw() != 16) { fprintf(stderr, "this code assumes 512-bit streaming vectors\n"); return 1; }

    size_t nn = (size_t)n * n;
    std::vector<float> A(nn), B(nn), C(nn), ref(nn);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, A.data(), n, B.data(), n,
                0.0f, ref.data(), n);

    const double flop = 2.0 * n * n * (double)n;
    for (int a = 2; a < argc; a++) {
        std::string name = argv[a];
        int K = 0, packA = 0;  // packA: 1 = pack A inside the timing, 2 = before it
        if (name.rfind("sme4_packA_k", 0) == 0) { K = atoi(name.c_str() + 12); packA = 1; }
        else if (name.rfind("sme4_prepacked_k", 0) == 0) { K = atoi(name.c_str() + 16); packA = 2; }
        else if (name.rfind("sme4_pack_k", 0) == 0) K = atoi(name.c_str() + 11);
        if (K) {
            if (K < 16 || K % 16 || n % K) {
                fprintf(stderr, "%s: K must be a multiple of 16 that divides n\n", name.c_str());
                return 1;
            }
        } else if (name != "sme1" && name != "sme4" && name != "accelerate" && name != "transpose") {
            fprintf(stderr, "unknown kernel %s\n", name.c_str());
            return 1;
        }

        if (packA) {
            g_Ap.resize(nn);
            if (packA == 2) pack_A(A.data(), g_Ap.data(), n);
        }
        double best = 1e30;
        for (int rep = 0; rep < 10; rep++) {
            std::fill(C.begin(), C.end(), 0.0f);
            double t0 = now();
            if (name == "sme1") mm_sme1(A.data(), B.data(), C.data(), n);
            else if (name == "sme4") mm_sme4(A.data(), B.data(), C.data(), n);
            else if (packA) {
                if (packA == 1) pack_A(A.data(), g_Ap.data(), n);
                multiply_packedA(g_Ap.data(), B.data(), C.data(), n, K);
            }
            else if (K) mm_sme4_pack(A.data(), B.data(), C.data(), n, K);
            else if (name == "transpose") { g_At.resize(nn); transpose(A.data(), g_At.data(), n); }
            else cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, A.data(), n,
                             B.data(), n, 0.0f, C.data(), n);
            best = std::min(best, now() - t0);
        }

        if (name == "transpose") {
            printf("TRANSPOSE,sme,%d,%.3f ms\n", n, best * 1e3);
            continue;
        }
        double err = 0;
        for (size_t i = 0; i < nn; i++) err = std::max(err, (double)std::fabs(C[i] - ref[i]));
        if (err > 1e-2) fprintf(stderr, "WARNING: %s n=%d max error %g\n", name.c_str(), n, err);
        printf("RESULT,sme,%d,%s,%.3f,%.1f\n", n, name.c_str(), best * 1e3, flop / best / 1e9);
        fflush(stdout);
    }
    return 0;
}
