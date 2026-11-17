CXX      = clang++
RF       = ../roofline/roofline_example

MACHINE := $(shell sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)
RESULTS  = results.M5.csv
RESULTS_MT = results_mt.M5.csv
CEILINGS = ceilings.M5.txt

TILES      = 4 8 16 32 64 128 256
TILED      = $(foreach T,$(TILES),tile_k$(T) tile_ijk$(T))
FAST_KERNS = ijk ijk_local ikj $(TILED)

# One source, three builds: the blog's first step is a change of flags.
all: mm_O0 mm_O3 mm_fast mm_mt

mm_O0: mm_bench.cpp
	$(CXX) -std=c++17 -DBUILD='"O0"' -o $@ $<

mm_O3: mm_bench.cpp
	$(CXX) -std=c++17 -O3 -march=native -DBUILD='"O3"' -o $@ $<

mm_fast: mm_bench.cpp
	$(CXX) -std=c++17 -O3 -march=native -ffast-math -DBUILD='"fast"' -o $@ $<

# Multithreaded: partition C, or partition A and B along k and merge.
mm_mt: mm_mt.cpp $(RF)/mt_common.hpp
	$(CXX) -std=c++17 -O3 -march=native -ffast-math -o $@ $<

# 1, 2 and 4 threads: the M5's performance cores. k_atomic (n^3 atomic adds)
# runs at n = 1024 only; at 2048 it is 8.6 billion of them.
$(RESULTS_MT): mm_mt
	rm -f $@
	for p in 1 2 4; do \
	    ./mm_mt 1024 $$p c_rows c_rows_tiled c_cols_tiled c_square_tiled c_tiles256 k_atomic k_atomic_row k_atomic_stagger k_private | tee -a $@; \
	    ./mm_mt 2048 $$p c_rows c_rows_tiled c_cols_tiled c_square_tiled c_tiles256 k_atomic_row k_atomic_stagger k_private | tee -a $@; \
	done

run_mt: $(RESULTS_MT)

# Single-core ceilings from the roofline examples.
$(CEILINGS):
	$(MAKE) -C $(RF) stream peak_flops
	$(RF)/stream     | tee $@
	$(RF)/peak_flops | tee -a $@

# n = 1024 is the blog's size; at n = 2048 the matrices no longer fit in
# the M5's 16 MB L2. The -O0 naive kernel at 2048 takes about a minute.
$(RESULTS): mm_O0 mm_O3 mm_fast
	rm -f $@
	for n in 1024 2048; do \
	    ./mm_O0   $$n ijk                 | tee -a $@; \
	    ./mm_O3   $$n ijk ikj             | tee -a $@; \
	    ./mm_fast $$n $(FAST_KERNS)       | tee -a $@; \
	done

run: $(RESULTS)

plot: $(RESULTS) $(CEILINGS)
	python3 plot_mm.py $(RESULTS) $(CEILINGS) --machine "$(MACHINE)"

clean:
	rm -f mm_O0 mm_O3 mm_fast mm_mt

.PHONY: all run run_mt plot clean
