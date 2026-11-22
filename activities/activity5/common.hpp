// common.hpp -- shared by every program in Activity 5. You should not need
// to change anything in this file.
//
//   sha384()      the SHA-384 digest of a block of bytes (FIPS 180-4)
//   Item          one file: its path and its contents
//   read_file()   read a whole file into an Item
//   Result        one file's path, size and digest
//   hash_item()   the consumer's work: Item -> Result
//   report()      group results by digest, print the duplicates summary,
//                 the run time and the peak memory

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <sys/resource.h>

namespace fs = std::filesystem;

// ---------------------------------------------------------------- SHA-384
//
// SHA-384 is SHA-512 with different initial values, keeping the first 384
// bits (48 bytes) of the result.

using Digest = std::array<uint8_t, 48>;

namespace sha512_detail {

static const uint64_t K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL,
};

static inline uint64_t rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

static inline uint64_t load_be64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v = (v << 8) | p[i];
    return v;
}

static inline void compress(uint64_t h[8], const uint8_t block[128]) {
    uint64_t w[80];
    for (int t = 0; t < 16; t++) w[t] = load_be64(block + 8 * t);
    for (int t = 16; t < 80; t++) {
        uint64_t s0 = rotr(w[t - 15], 1) ^ rotr(w[t - 15], 8) ^ (w[t - 15] >> 7);
        uint64_t s1 = rotr(w[t - 2], 19) ^ rotr(w[t - 2], 61) ^ (w[t - 2] >> 6);
        w[t] = w[t - 16] + s0 + w[t - 7] + s1;
    }
    uint64_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
    for (int t = 0; t < 80; t++) {
        uint64_t S1 = rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41);
        uint64_t ch = (e & f) ^ (~e & g);
        uint64_t t1 = k + S1 + ch + K[t] + w[t];
        uint64_t S0 = rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39);
        uint64_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint64_t t2 = S0 + maj;
        k = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += k;
}

}  // namespace sha512_detail

static inline Digest sha384(const uint8_t* data, size_t len) {
    using namespace sha512_detail;
    uint64_t h[8] = {
        0xcbbb9d5dc1059ed8ULL, 0x629a292a367cd507ULL, 0x9159015a3070dd17ULL, 0x152fecd8f70e5939ULL,
        0x67332667ffc00b31ULL, 0x8eb44a8768581511ULL, 0xdb0c2e0d64f98fa7ULL, 0x47b5481dbefa4fa4ULL,
    };
    size_t full = len / 128;
    for (size_t i = 0; i < full; i++) compress(h, data + 128 * i);

    // Padding: the leftover bytes, a 1 bit, zeros, and the length in bits as
    // a 128-bit big-endian number, filling one or two final blocks.
    uint8_t tail[256] = {0};
    size_t rem = len - 128 * full;
    std::memcpy(tail, data + 128 * full, rem);
    tail[rem] = 0x80;
    size_t tail_len = (rem + 1 + 16 <= 128) ? 128 : 256;
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) tail[tail_len - 1 - i] = (uint8_t)(bits >> (8 * i));
    for (size_t off = 0; off < tail_len; off += 128) compress(h, tail + off);

    Digest out;
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 8; j++) out[8 * i + j] = (uint8_t)(h[i] >> (56 - 8 * j));
    return out;
}

static inline std::string to_hex(const Digest& d) {
    static const char* hex = "0123456789abcdef";
    std::string s(96, ' ');
    for (int i = 0; i < 48; i++) {
        s[2 * i] = hex[d[i] >> 4];
        s[2 * i + 1] = hex[d[i] & 15];
    }
    return s;
}

// ---------------------------------------------------------------- files

struct Item {
    std::string path;
    std::vector<uint8_t> data;
};

static inline Item read_file(const fs::path& path) {
    Item item;
    item.path = path.string();
    std::ifstream in(path, std::ios::binary);
    in.seekg(0, std::ios::end);
    item.data.resize((size_t)in.tellg());
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(item.data.data()), (std::streamsize)item.data.size());
    return item;
}

struct Result {
    std::string path;
    uint64_t bytes = 0;
    Digest digest{};
};

static inline Result hash_item(const Item& item) {
    Result r;
    r.path = item.path;
    r.bytes = item.data.size();
    r.digest = sha384(item.data.data(), item.data.size());
    return r;
}

// ---------------------------------------------------------------- report

static inline double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static inline double peak_rss_mb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return ru.ru_maxrss / 1048576.0;  // bytes on macOS
#else
    return ru.ru_maxrss / 1024.0;     // kilobytes on Linux
#endif
}

// Groups the results by digest and prints:
//   files, bytes        everything that was hashed
//   unique              distinct contents
//   duplicate groups    contents that appear more than once
//   reclaimable         bytes you would free by keeping one copy of each
//   fingerprint         a digest of the sorted list of digests: the same for
//                       every correct run on the same data, whatever order
//                       the files were hashed in
// and, with -v, every duplicate group with its paths.
static inline void report(std::vector<Result> results, double seconds, bool verbose = false) {
    std::sort(results.begin(), results.end(), [](const Result& a, const Result& b) {
        return a.digest != b.digest ? a.digest < b.digest : a.path < b.path;
    });
    uint64_t bytes = 0, reclaimable = 0;
    size_t unique = 0, groups = 0;
    std::string all;
    for (size_t i = 0; i < results.size();) {
        size_t j = i;
        while (j < results.size() && results[j].digest == results[i].digest) j++;
        unique++;
        if (j - i > 1) {
            groups++;
            reclaimable += (j - i - 1) * results[i].bytes;
            if (verbose) {
                printf("%s\n", to_hex(results[i].digest).substr(0, 16).c_str());
                for (size_t x = i; x < j; x++) printf("    %s\n", results[x].path.c_str());
            }
        }
        for (size_t x = i; x < j; x++) {
            bytes += results[x].bytes;
            all.append(reinterpret_cast<const char*>(results[x].digest.data()), results[x].digest.size());
        }
        i = j;
    }
    Digest fp = sha384(reinterpret_cast<const uint8_t*>(all.data()), all.size());
    printf("files             %zu\n", results.size());
    printf("bytes             %.1f MB\n", bytes / 1048576.0);
    printf("unique            %zu\n", unique);
    printf("duplicate groups  %zu\n", groups);
    printf("reclaimable       %.1f MB\n", reclaimable / 1048576.0);
    printf("fingerprint       %s\n", to_hex(fp).substr(0, 32).c_str());
    printf("time              %.3f s\n", seconds);
    printf("peak memory       %.0f MB\n", peak_rss_mb());
}
