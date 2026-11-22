// serial.cpp -- the reference answer for Activity 5: walk every directory,
// hash every file, one at a time. make check compares each part's
// fingerprint with this one's.
//
//   ./serial [-v] <dir> <dir> ...

#include "common.hpp"

int main(int argc, char** argv) {
    bool verbose = false;
    std::vector<std::string> dirs;
    for (int a = 1; a < argc; a++) {
        if (std::string(argv[a]) == "-v") verbose = true;
        else dirs.push_back(argv[a]);
    }
    if (dirs.empty()) {
        fprintf(stderr, "usage: %s [-v] <dir> <dir> ...\n", argv[0]);
        return 1;
    }
    std::vector<Result> results;
    double t0 = now();
    for (const auto& d : dirs)
        for (const auto& entry : fs::recursive_directory_iterator(d))
            if (entry.is_regular_file()) results.push_back(hash_item(read_file(entry.path())));
    double t1 = now();
    report(results, t1 - t0, verbose);
    return 0;
}
