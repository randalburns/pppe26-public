// part1_cilk.cpp -- Activity 5, Part 1(b): one producer, many consumers,
// with Cilk and no buffer between them.
//
//   ./part1_cilk <dir> [-v]          workers: CILK_NWORKERS
//
// As given, this runs serially: the producer hashes each file itself, one at
// a time. Your job is to make each consume() run in parallel with the
// producer and with the other consumers.

#include "common.hpp"

#include <deque>

// TODO 1(b): include <cilk/cilk.h>

// The consumer: hash one file. Each call writes only its own Result, so
// calls for different files share nothing.
static void consume(Item* item, Result* out) {
    *out = hash_item(*item);
    delete item;
}

// The producer: walk dir recursively, read each file, and hand it to a
// consumer. results grows by one slot per file. A deque never moves the slots
// it already holds, so a consumer can fill its slot while the producer keeps
// adding new ones.
static void producer(const std::string& dir, std::deque<Result>& results) {
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        Item* item = new Item(read_file(entry.path()));
        results.emplace_back();
        Result* slot = &results.back();
        consume(item, slot);  // TODO 1(b): spawn this call
    }
    // TODO 1(b): wait for every consumer this producer spawned
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <dir> [-v]\n", argv[0]);
        return 1;
    }
    bool verbose = argc > 2 && std::string(argv[2]) == "-v";
    std::deque<Result> results;

    double t0 = now();
    // TODO 1(b): nothing to add here, unless you want to
    producer(argv[1], results);
    double t1 = now();

    report(std::vector<Result>(results.begin(), results.end()), t1 - t0, verbose);
    return 0;
}
