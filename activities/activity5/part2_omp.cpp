// part2_omp.cpp -- Activity 5, Part 2(a): many producers, many consumers,
// with OpenMP and a bounded circular buffer between them.
//
//   ./part2_omp [-c consumers] [-b capacity] [-v] <dir> <dir> ...
//
// One producer per directory; -c consumers (default 4); a buffer that holds
// -b files (default 16). Threads: one per producer and one per consumer.
//
// As given, this runs serially and the buffer is unsynchronized: every
// producer runs to the end, then every consumer. That only works when the
// buffer can hold every file at once (try -b 100000). Your version must run
// the producers and consumers at the same time and work at -b 16.

#include "common.hpp"

#include <cstdlib>

// TODO 2(a): include <omp.h>

// A bounded first-in, first-out buffer of files, stored in a circular array:
// slots[head] is the oldest item, and the next put goes count slots after it.
//
// TODO 2(a): make put, take and close safe to call from many threads at
// once. put must wait while the buffer is full; take must wait while it is
// empty and not yet closed.
struct CircularBuffer {
    explicit CircularBuffer(size_t capacity) : slots(capacity) {}

    // Add an item at the tail.
    void put(Item&& item) {
        if (count == slots.size()) {
            fprintf(stderr, "buffer full: the serial starter needs -b >= the number of files\n");
            exit(1);
        }
        slots[(head + count) % slots.size()] = std::move(item);
        count++;
    }

    // Remove the item at the head. Returns false when there will never be
    // another item: the buffer is empty and closed.
    bool take(Item& item) {
        if (count == 0) return false;
        item = std::move(slots[head]);
        head = (head + 1) % slots.size();
        count--;
        return true;
    }

    // No more items will be put. Call it once, after the last producer has
    // finished; consumers then drain what is left and stop.
    void close() { closed = true; }

    std::vector<Item> slots;
    size_t head = 0, count = 0;
    bool closed = false;
};

// The producer: walk dir recursively and put each file in the buffer.
static void producer(const std::string& dir, CircularBuffer& buf) {
    for (const auto& entry : fs::recursive_directory_iterator(dir))
        if (entry.is_regular_file()) buf.put(read_file(entry.path()));
}

// The consumer: take files until the buffer is closed and empty, and hash
// each one. Each consumer has its own result list, so the only data the
// consumers share is the buffer.
static void consumer(CircularBuffer& buf, std::vector<Result>& out) {
    Item item;
    while (buf.take(item)) out.push_back(hash_item(item));
}

int main(int argc, char** argv) {
    int consumers = 4;
    size_t capacity = 16;
    bool verbose = false;
    std::vector<std::string> dirs;
    for (int a = 1; a < argc; a++) {
        std::string s = argv[a];
        if (s == "-c" && a + 1 < argc) consumers = atoi(argv[++a]);
        else if (s == "-b" && a + 1 < argc) capacity = (size_t)atol(argv[++a]);
        else if (s == "-v") verbose = true;
        else dirs.push_back(s);
    }
    if (dirs.empty() || consumers < 1 || capacity < 1) {
        fprintf(stderr, "usage: %s [-c consumers] [-b capacity] [-v] <dir> <dir> ...\n", argv[0]);
        return 1;
    }

    CircularBuffer buf(capacity);
    std::vector<std::vector<Result>> out(consumers);  // one list per consumer

    double t0 = now();
    // TODO 2(a): run every producer and every consumer at the same time, each on
    // its own thread, and close the buffer once every producer has finished
    for (const auto& d : dirs) producer(d, buf);
    buf.close();
    for (int c = 0; c < consumers; c++) consumer(buf, out[c]);
    double t1 = now();

    std::vector<Result> all;
    for (auto& o : out) all.insert(all.end(), o.begin(), o.end());
    report(all, t1 - t0, verbose);
    return 0;
}
