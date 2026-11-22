# Activity 5 -- producers, consumers and a circular buffer.
#
#   make data      generate the data set (about 770 MB, in ./data)
#   make           build serial and the four parts
#   make check-setup  verify the starter builds and runs
#   make check        verify fingerprints and check the index on 4 threads
#   make bench     Part 3: the index's insert rate at each shard and thread count

OPENCILK ?= $(HOME)/opencilk
CILK_CXX ?= $(OPENCILK)/bin/clang++
CXXFLAGS = -std=c++17 -O3 -Wall

UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
OMP_PREFIX ?= $(shell brew --prefix libomp)
OMP = -Xpreprocessor -fopenmp -I"$(OMP_PREFIX)/include" -L"$(OMP_PREFIX)/lib" -Wl,-rpath,"$(OMP_PREFIX)/lib" -lomp
else
OMP = -fopenmp
endif
CILK = -fopencilk

DATA = data
DIRS = $(sort $(wildcard $(DATA)/d*))

BINS = serial part1_omp part1_cilk part2_omp index_bench

all: $(BINS)

serial: serial.cpp common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

part1_omp part2_omp: %: %.cpp common.hpp
	$(CXX) $(CXXFLAGS) $(OMP) -o $@ $<

part1_cilk: %: %.cpp common.hpp
	"$(CILK_CXX)" $(CXXFLAGS) $(CILK) -o $@ $<

index_bench: index_bench.cpp index_omp.hpp common.hpp
	$(CXX) $(CXXFLAGS) $(OMP) -o $@ $<

data:
	python3 gen_data.py "$(DATA)"

# Part 3: insert rate of the index at each shard count and thread count.
SHARDS  = 1 4 8 16 32
THREADS = 1 2 4 8
bench: index_bench
	@set -e; for s in $(SHARDS); do for t in $(THREADS); do \
	    OMP_NUM_THREADS=$$t ./index_bench -s $$s; \
	done; done

# The starter index is only safe on one thread. Use check after implementing it.
check-setup: INDEX_THREADS = 1
check-setup: BUFFER_SIZE = 100000
check: INDEX_THREADS = 4
check: BUFFER_SIZE = 16
check check-setup: CHECK_BINS = part1_omp part1_cilk part2_omp index_bench
check check-setup: all

# Part 2 runs one producer per top-level directory.
check check-setup:
	@test -n "$(DIRS)" || { echo "No data. Run make data first." >&2; exit 1; }
	@set -e; \
	    report=$$(./serial $(DIRS)); \
	    expected=$$(printf '%s\n' "$$report" | grep '^fingerprint '); \
	    set -- $(CHECK_BINS); \
	    for program in "$$1" "$$2" "$$3"; do \
	        printf '%-20s' "$$program"; \
	        case $$program in \
	            part2_omp*) report=$$(OMP_DYNAMIC=FALSE ./$$program -c 4 -b $(BUFFER_SIZE) $(DIRS)) ;; \
	            *) report=$$(OMP_NUM_THREADS=2 CILK_NWORKERS=2 ./$$program "$(DATA)") ;; \
	        esac; \
	        actual=$$(printf '%s\n' "$$report" | grep '^fingerprint '); \
	        if [ "$$actual" != "$$expected" ]; then echo WRONG; exit 1; fi; \
	        echo ok; \
	    done; \
	    OMP_NUM_THREADS=$(INDEX_THREADS) ./$$4 -s 16 -n 100000

clean:
	rm -f $(BINS)

.PHONY: all data check check-setup bench clean
