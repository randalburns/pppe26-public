// index_omp.hpp -- Activity 5, Part 3: the deduplication index.
//
// A hash table from digest to storage location, split into S shards. Each
// shard is an independent table; a digest always goes to the same shard.
//
// The digests are SHA-384 outputs, so their bits are already uniformly
// random: the first byte picks the shard, and bytes 8-15 are the hash
// within the shard. No second hash function is needed, and the load
// spreads evenly over the shards by construction.
//
// TODO 3(a): as given, the index is not safe to use from more than one
// thread. Protect it with critical sections so that operations on
// different shards can run at the same time, and operations on the same
// shard cannot.

#pragma once

#include "common.hpp"

#include <unordered_map>

struct DigestHash {
    size_t operator()(const Digest& d) const {
        uint64_t h;
        std::memcpy(&h, d.data() + 8, sizeof h);  // byte 0 picks the shard
        return (size_t)h;
    }
};

class ShardedIndex {
public:
    // shards: S. expected: about how many distinct digests will be inserted,
    // so each shard can size its table once instead of growing it.
    ShardedIndex(int shards, size_t expected) : shards_(shards) {
        for (auto& s : shards_) s.map.reserve(expected / shards + 1);
    }

    // Insert-if-absent. If digest is new, record where it is stored and
    // return true. If it is already present, leave the index unchanged and
    // return false: this upload is a duplicate.
    bool insert(const Digest& digest, uint64_t location) {
        Shard& s = shards_[shard_of(digest)];
        return s.map.emplace(digest, location).second;
    }

    // Where the copy of digest is stored, or -1 if it has never been seen.
    int64_t find(const Digest& digest) {
        Shard& s = shards_[shard_of(digest)];
        auto it = s.map.find(digest);
        return it == s.map.end() ? -1 : (int64_t)it->second;
    }

    // Distinct digests in the index. Call it when no thread is inserting.
    size_t size() const {
        size_t n = 0;
        for (const auto& s : shards_) n += s.map.size();
        return n;
    }

private:
    struct Shard {
        std::unordered_map<Digest, uint64_t, DigestHash> map;
    };

    size_t shard_of(const Digest& digest) const { return digest[0] % shards_.size(); }

    std::vector<Shard> shards_;
};
