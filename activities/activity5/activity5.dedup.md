# Activity 5 — A Deduplicating Upload Service: Producers and Consumers

## Background

A storage service accepts uploaded files from many clients, and many of the
uploads are copies of files it already holds: the same photo shared by a
hundred people, the same library vendored into a thousand projects. A
**deduplicating** service stores each distinct file once. It computes a
**signature** of every upload, and two uploads with the same signature are
stored only once.

The signature is a SHA-384 digest of the file's contents, the usual choice
for content digests. SHA-384 is a cryptographic hash: it is deterministic
(the same bytes always give the same 48-byte digest), and finding two
different files with the same digest is infeasible, so in practice equal
digests mean equal files.

Ingesting an upload is two kinds of work:

  * **receiving** it: reading the file's bytes into memory (I/O)
  * **signing** it: computing its SHA-384 digest (CPU, about 0.4 GB/s on one
    core of an M5)

That is the producer–consumer pattern from Lecture 15. **Producers** receive
uploads and hand them over; **consumers** sign them. This activity builds the
ingest pipeline first with no buffer between producer and consumers, in
OpenMP and in Cilk (Part 1), then with a bounded circular buffer in OpenMP
(Part 2). In those parts, each consumer just computes digests, and the
report finds the duplicates at the end. Part 3 builds the shared index that
detects duplicates as they arrive.

## Setup

Use Python 3.9 or newer, Make, a C++17 compiler, and the course's
[OpenCilk 3.0 toolchain](https://www.opencilk.org/doc/users-guide/install/).
The Makefile looks for OpenCilk at `~/opencilk/bin/clang++`. For a different
installation directory, pass `OPENCILK=/path/to/opencilk` to Make.

- **Apple silicon macOS.** Install the Xcode Command Line Tools with
  `xcode-select --install` if needed, and install Homebrew's OpenMP runtime
  with `brew install libomp`. The Makefile locates it with `brew --prefix libomp`.
  For a separate OpenMP installation, pass `OMP_PREFIX=/path/to/libomp` to Make.
- **Ubuntu 24.04 x86-64 or WSL2.** Run
  `sudo apt update && sudo apt install build-essential python3`, then install
  the Ubuntu OpenCilk toolchain. The OpenMP programs use GCC and its OpenMP
  runtime. OpenCilk builds the Cilk program.

Run these commands from the directory containing the assignment's Makefile.
Allow about 800 MB of disk space for the uploads, in addition to the toolchain.
The serial Part 2 starter holds the full dataset in memory during the setup check.

```bash
make data           # run once to generate the default dataset in ./data
make check-setup    # build and verify the starter programs
./serial data/d*    # print the baseline report
```

The generator requires a new output directory. If `data/` already contains
the complete dataset, reuse it. To generate another copy, use a new directory,
for example `make data DATA=data-new`, and pass `DATA=data-new` to the Make
checks that use it.

`data/d0` … `data/d7` are eight clients' upload directories, each with a few
levels of subdirectories. The serial baseline and the upload programs in
Parts 1 and 2 print reports in this format:

```
files             836
bytes             772.7 MB
unique            632
duplicate groups  150
reclaimable       142.3 MB
fingerprint       437b7949cd28415c5a1792b2b17ba838
time              1.600 s
peak memory       42 MB
```

`reclaimable` is the storage deduplication saves. `fingerprint` is a digest
of every file's digest, sorted: the same for every correct run, in whatever
order the files were signed. `peak memory` is the most memory the process
held at once. `make check-setup` compares each fingerprint with the serial
baseline, gives the serial Part 2 starter a large buffer, and checks the index
with one thread. After implementing all three parts, run `make check` to
check the fingerprints with a 16-file buffer in Part 2 and test the index
with four threads.

### The code

| file | what it is |
|---|---|
| `common.hpp` | SHA-384, `Item` (a file's path and bytes), `read_file`, `hash_item`, `report`. Don't change it |
| `serial.cpp` | the reference: one file at a time |
| `part1_omp.cpp`, `part1_cilk.cpp` | Part 1 starters |
| `part2_omp.cpp` | Part 2 starter |
| `index_omp.hpp`, `index_bench.cpp` | Part 3: the deduplication index and its benchmark |

Every starter runs serially as given, and gets the right answer. The producer
and consumer functions are written for you. Your job is the parallelism and
the synchronization.

## Part 1 — one producer, no buffer

One producer walks the whole upload tree (`./part1_omp data`) recursively.
For each file, it reads the bytes and hands them to a consumer:

```cpp
static void producer(const std::string& dir, std::deque<Result>& results) {
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        Item* item = new Item(read_file(entry.path()));
        results.emplace_back();
        Result* slot = &results.back();
        consume(item, slot);
    }
}
```

There is no buffer: each file becomes its own unit of work. Each call to
`consume` writes only its own `Result` slot, so different calls share
nothing.

**1(a)** In `part1_omp.cpp`, use OpenMP tasks: one thread runs the producer,
and each `consume` call becomes a task that any thread in the team can run.
Do not add a buffer. Check the fingerprints with `make check-setup`.

**1(b)** Do the same in `part1_cilk.cpp` with `cilk_spawn` and `cilk_sync`.

Measure both, and `serial` for comparison. With 2 threads, one thread
produces and one consumes, so the producer is much faster than its consumers:

| | threads | time | peak memory |
|---|---|---|---|
| `serial` | 1 | | |
| `part1_omp` | `OMP_NUM_THREADS=2` | | |
| `part1_omp` | all cores | | |
| `part1_cilk` | `CILK_NWORKERS=2` | | |
| `part1_cilk` | all cores | | |

**1(c)** What hazard exists in this design? Consider the case when the
producer is much faster than the consumers.

## Part 2 — many producers, many consumers, a bounded circular buffer

Now each producer is one client: it gets one upload directory and walks it
recursively.

```bash
./part2_omp -c 4 -b 16 data/d*     # 8 producers, 4 consumers, a 16-file buffer
```

> **A simplifying assumption.** The data set has a fixed number of clients,
> eight directories, and the program starts exactly one producer for each.
> A real upload service can't do that: the number of clients is set by the
> outside world, and every producer costs a thread and holds a file in
> memory while it waits for room in the buffer. It would instead run a
> small, fixed pool of producers, chosen for the machine, that take the next
> pending upload from a list as they finish one. Matching producers to
> directories keeps this activity about the buffer.

Between them is a bounded circular buffer: an array of `-b` slots, with
`slots[head]` the oldest item and the next put going `count` slots after it.
These are the producer and consumer, as given:

```cpp
static void producer(const std::string& dir, CircularBuffer& buf) {
    for (const auto& entry : fs::recursive_directory_iterator(dir))
        if (entry.is_regular_file()) buf.put(read_file(entry.path()));
}

static void consumer(CircularBuffer& buf, std::vector<Result>& out) {
    Item item;
    while (buf.take(item)) out.push_back(hash_item(item));
}
```

`take` returns `false` when there will never be another item: the buffer is
empty and closed. `close` is called once, after the last producer finishes.
Each consumer appends to its own result list, so the only data the threads
share is the buffer, and the buffer as given is **not synchronized**. The
serial starter runs every producer and then every consumer, which only works
if the buffer can hold every file at once (`-b 100000`).

Part 2 uses OpenMP only. A bounded buffer makes producers and consumers
*block* on each other, which is a poor fit for Cilk.

Your version must run all producers and all consumers at the same time, and
work at `-b 16`:

  * `put` must wait while the buffer is full
  * `take` must wait while the buffer is empty and not yet closed
  * the buffer is closed once every producer has finished, and not before

**2(a)** Implement it in `part2_omp.cpp` with OpenMP. Give the buffer an
`omp_lock_t` member, and use it to protect `put`, `take` and `close`.
When `put` finds the buffer full, or `take` finds it empty and not closed,
unlock, call `std::this_thread::yield()` to let other threads run, and try
again. A `put` or `take` that succeeds just unlocks and returns.

Compare the fingerprint from the 16-file-buffer command above with the serial
baseline, then measure:

| | consumers | buffer | time | peak memory |
|---|---|---|---|---|
| `part2_omp` | 1 | 16 | | |
| `part2_omp` | 4 | 16 | | |
| `part2_omp` | 1 | 1000 | | |

**2(b)** How does this design address the hazard of Part 1?

## Part 3 — the deduplication index

So far, duplicates are found after the fact: the report groups every result
by digest at the end. The service has to know *while* it ingests: given a
new upload's digest, has it stored this file before? If so, it records a
reference to the existing copy instead of storing the bytes again.

That is one operation on a shared table, **insert-if-absent**: if the
digest is new, record where its file is stored and say so; if it is already
there, say it's a duplicate. Every consumer does it for every file, so the
table is shared, written by every thread, and has to be exact: a missed
duplicate wastes storage, and a false duplicate loses a file.

`index_omp.hpp` gives you the table, a `ShardedIndex` split into S **shards**.
Each shard is an independent hash table, and a digest always goes to the
same shard:

```cpp
bool insert(const Digest& digest, uint64_t location) {
    Shard& s = shards_[shard_of(digest)];
    return s.map.emplace(digest, location).second;   // true if it was new
}
size_t shard_of(const Digest& digest) const { return digest[0] % shards_.size(); }
```

A SHA-384 digest is uniformly random, so its first byte spreads the digests
evenly over the shards, and bytes 8–15 serve as the hash inside each shard:
no second hash function is needed.

`index_bench.cpp` measures the table on its own: the threads split a
list of 4 million digests, a quarter of them repeats, and the benchmark reports
millions of inserts per second. It also checks the answer: each distinct
digest must be reported new exactly once.

After adding the locks, run:

```bash
OMP_NUM_THREADS=4 ./index_bench -s 16     # 16 shards, 4 threads
make bench                                # every S in 1 4 8 16 32, 1 to 8 threads
```

To run the starter benchmark before adding the locks, use
`OMP_NUM_THREADS=1 ./index_bench -s 16`.

**3(a)** Protect the index in `index_omp.hpp` with OpenMP locks
(`omp_lock_t`), so that operations on *different* shards can run at the
same time and operations on the *same* shard cannot.

**3(b)** Run `make bench` and fill in the insert rate, in millions per
second:

| S \ threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 | | | | |
| 4 | | | | |
| 8 | | | | |
| 16 | | | | |
| 32 | | | | |

**3(c)** With S = 1, what happens to the insert rate as you add threads,
and why?

**3(d)** Why do more shards help, and why does the improvement level off?

**3(e)** Why does protecting the shards with `#pragma omp critical` not
meet the goal of parallel access to shards? What would happen at runtime?

## What to hand in

  * `part1_omp.cpp`, `part1_cilk.cpp`, `part2_omp.cpp` and `index_omp.hpp`,
    each passing `make check`
  * your tables, and the machine you ran them on
  * your answers to 1(c), 2(b), 3(c), 3(d) and 3(e)
