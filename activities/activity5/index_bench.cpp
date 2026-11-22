// index_bench.cpp -- Activity 5, Part 3: how fast is the deduplication index?
//
//   ./index_bench [-s shards] [-n inserts] [-d duplicate-fraction]
//   threads: OMP_NUM_THREADS
//
// Inserts n digests into a ShardedIndex from every thread at once, and
// reports millions of inserts per second. A fraction d of the inserts
// (default 0.25) repeat a digest inserted earlier, as duplicate uploads do.
//
// The digests are random 48-byte strings rather than SHA-384s of real data:
// a SHA-384 digest is indistinguishable from random bytes, and generating
// them directly keeps the benchmark about the index, not about hashing.
//
// Correctness check: exactly one insert of each distinct digest may return
// true. If the count of true returns, or the size of the index, differs
// from the number of distinct digests, the index lost or duplicated an
// insert, and the benchmark says so.
//
// You should not need to change this file.

#include "index_omp.hpp"

#include <omp.h>

#include <cstdlib>
#include <random>

int main(int argc, char** argv) {
    int shards = 16;
    size_t n = 4'000'000;
    double dup = 0.25;
    for (int a = 1; a + 1 < argc; a += 2) {
        std::string s = argv[a];
        if (s == "-s") shards = atoi(argv[a + 1]);
        else if (s == "-n") n = (size_t)atol(argv[a + 1]);
        else if (s == "-d") dup = atof(argv[a + 1]);
    }

    // unique distinct digests, then n inserts: each distinct digest once,
    // plus duplicates drawn from them, in random order.
    size_t unique = (size_t)(n * (1.0 - dup));
    std::mt19937_64 rng(5);
    std::vector<Digest> pool(unique);
    for (auto& d : pool)
        for (size_t i = 0; i < d.size(); i += 8) {
            uint64_t v = rng();
            std::memcpy(d.data() + i, &v, 8);
        }
    std::vector<uint32_t> order(n);
    for (size_t i = 0; i < n; i++) order[i] = (uint32_t)(i < unique ? i : rng() % unique);
    std::shuffle(order.begin(), order.end(), rng);

    ShardedIndex index(shards, unique);
    size_t fresh = 0;
    double t0 = now();
#pragma omp parallel for schedule(static) reduction(+ : fresh)
    for (size_t i = 0; i < n; i++)
        if (index.insert(pool[order[i]], i)) fresh++;
    double t1 = now();
    int threads = omp_get_max_threads();
    bool ok = fresh == unique && index.size() == unique;
    printf("RESULT_INDEX,%d,%d,%zu,%.4f,%.2f,%s\n", shards, threads, n, t1 - t0,
           n / (t1 - t0) / 1e6, ok ? "ok" : "WRONG");
    if (!ok)
        fprintf(stderr, "WRONG: %zu inserts returned true and the index holds %zu digests; "
                        "there are %zu distinct digests\n", fresh, index.size(), unique);
    return ok ? 0 : 1;
}
