CXX      = clang++
CXXFLAGS = -std=c++17 -O3 -march=native

OPENBLAS = /opt/homebrew/opt/openblas

# Machine description for the plot titles, read from the host rather than
# hardcoded. CORES is the P/E split on heterogeneous Apple Silicon (empty
# where sysctl has no second perf level).
MACHINE := $(shell sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)
CORES   := $(shell p=$$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null); \
                   e=$$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null); \
                   [ -n "$$e" ] && echo "$${p}P+$${e}E")

all: stream peak_flops sort_bench dot_bench saxpy_bench stencil_bench spmv_bench nbody_bench matmul_naive_bench matmul_bench openblas_bench

all_mt: stream_mt peak_flops_mt sort_bench_mt dot_bench_mt saxpy_bench_mt stencil_bench_mt spmv_bench_mt nbody_bench_mt matmul_naive_bench_mt

stream: stream.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

peak_flops: peak_flops.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

sort_bench: sort_bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

dot_bench: dot_bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

saxpy_bench: saxpy_bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

stencil_bench: stencil_bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

spmv_bench: spmv_bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

nbody_bench: nbody_bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

matmul_naive_bench: matmul_naive_bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

matmul_bench: matmul_bench.cpp
	$(CXX) $(CXXFLAGS) -DACCELERATE_NEW_LAPACK -framework Accelerate -o $@ $<

openblas_bench: matmul_bench.cpp
	$(CXX) $(CXXFLAGS) -DUSE_OPENBLAS \
	    -I$(OPENBLAS)/include -L$(OPENBLAS)/lib -lopenblas \
	    -o $@ $<

# Multicore companions. All use std::thread (mt_common.hpp), no OpenMP/TBB
# dependency, consistent with the rest of this directory avoiding libraries.
stream_mt: stream_mt.cpp mt_common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

peak_flops_mt: peak_flops_mt.cpp mt_common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

sort_bench_mt: sort_bench_mt.cpp mt_common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

dot_bench_mt: dot_bench_mt.cpp mt_common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

saxpy_bench_mt: saxpy_bench_mt.cpp mt_common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

stencil_bench_mt: stencil_bench_mt.cpp mt_common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

spmv_bench_mt: spmv_bench_mt.cpp mt_common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

nbody_bench_mt: nbody_bench_mt.cpp mt_common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

matmul_naive_bench_mt: matmul_naive_bench_mt.cpp mt_common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Run all benchmarks, capture results, generate the plot.
# sgemm (Accelerate/OpenBLAS) is deliberately excluded: on Apple Silicon it
# dispatches through the AMX matrix coprocessor, not the NEON ALU pipeline
# that peak_flops.cpp ceilings model, so its points aren't bound by (and
# routinely blow past) this roofline's compute ceilings.
plot: all
	@echo "=== Memory Bandwidth (STREAM) ==="
	@./stream | tee /tmp/stream_out.txt
	@echo ""
	@echo "=== Peak FLOPS (SIMD FMA, SIMD no-FMA, Scalar FMA, Scalar no-FMA) ==="
	@./peak_flops | tee /tmp/flops_out.txt
	@echo ""
	@echo "=== std::sort 16M floats ==="
	@./sort_bench | tee /tmp/sort_out.txt
	@echo ""
	@echo "=== dot product 32M floats ==="
	@./dot_bench | tee /tmp/dot_out.txt
	@echo ""
	@echo "=== saxpy 32M floats ==="
	@./saxpy_bench | tee /tmp/saxpy_out.txt
	@echo ""
	@echo "=== 7-point stencil 256^3 ==="
	@./stencil_bench | tee /tmp/stencil_out.txt
	@echo ""
	@echo "=== SpMV 4M rows x 7 nnz/row ==="
	@./spmv_bench | tee /tmp/spmv_out.txt
	@echo ""
	@echo "=== N-body 4096 particles ==="
	@./nbody_bench | tee /tmp/nbody_out.txt
	@echo ""
	@echo "=== Naive sgemm 512x512, no BLAS/AMX (scalar/scalar-FMA/SIMD/SIMD-FMA) ==="
	@./matmul_naive_bench | tee /tmp/matmul_naive_out.txt
	@BW=$$(grep      "^PEAK_BANDWIDTH="            /tmp/stream_out.txt | cut -d= -f2); \
	 GF_SFMA=$$(grep "^PEAK_GFLOPS_SIMD_FMA="     /tmp/flops_out.txt  | cut -d= -f2); \
	 GF_SMUL=$$(grep "^PEAK_GFLOPS_SIMD_NOFMA="   /tmp/flops_out.txt  | cut -d= -f2); \
	 GF_SCFMA=$$(grep "^PEAK_GFLOPS_SCALAR_FMA="   /tmp/flops_out.txt  | cut -d= -f2); \
	 GF_SCMUL=$$(grep "^PEAK_GFLOPS_SCALAR_NOFMA=" /tmp/flops_out.txt  | cut -d= -f2); \
	 SORT_AI=$$(grep  "^SORT_AI="                   /tmp/sort_out.txt   | cut -d= -f2); \
	 SORT_GF=$$(grep  "^SORT_GFLOPS="               /tmp/sort_out.txt   | cut -d= -f2); \
	 DOT_AI=$$(grep   "^DOTPROD_AI="                /tmp/dot_out.txt    | cut -d= -f2); \
	 DOT_GF=$$(grep   "^DOTPROD_GFLOPS="            /tmp/dot_out.txt    | cut -d= -f2); \
	 SAXPY_AI=$$(grep "^SAXPY_AI="                  /tmp/saxpy_out.txt  | cut -d= -f2); \
	 SAXPY_GF=$$(grep "^SAXPY_GFLOPS="              /tmp/saxpy_out.txt  | cut -d= -f2); \
	 STEN_AI=$$(grep  "^STENCIL_AI="                /tmp/stencil_out.txt| cut -d= -f2); \
	 STEN_GF=$$(grep  "^STENCIL_GFLOPS="            /tmp/stencil_out.txt| cut -d= -f2); \
	 SPMV_AI=$$(grep  "^SPMV_AI="                   /tmp/spmv_out.txt   | cut -d= -f2); \
	 SPMV_GF=$$(grep  "^SPMV_GFLOPS="               /tmp/spmv_out.txt   | cut -d= -f2); \
	 NBODY_AI=$$(grep "^NBODY_AI="                  /tmp/nbody_out.txt  | cut -d= -f2); \
	 NBODY_GF=$$(grep "^NBODY_GFLOPS="              /tmp/nbody_out.txt  | cut -d= -f2); \
	 MM_AI=$$(grep    "^MATMUL_AI="                 /tmp/matmul_naive_out.txt | cut -d= -f2); \
	 MM_SC=$$(grep    "^MATMUL_SCALAR_GFLOPS="      /tmp/matmul_naive_out.txt | cut -d= -f2); \
	 MM_SCF=$$(grep   "^MATMUL_SCALAR_FMA_GFLOPS="  /tmp/matmul_naive_out.txt | cut -d= -f2); \
	 MM_SI=$$(grep    "^MATMUL_SIMD_GFLOPS="        /tmp/matmul_naive_out.txt | cut -d= -f2); \
	 MM_SIF=$$(grep   "^MATMUL_SIMD_FMA_GFLOPS="    /tmp/matmul_naive_out.txt | cut -d= -f2); \
	 python3 roofline.py "$${BW}" "$${GF_SFMA}" "$${GF_SMUL}" "$${GF_SCFMA}" "$${GF_SCMUL}" --machine "$(MACHINE)" \
	     --point "std::sort 16M"      "$${SORT_AI}"    "$${SORT_GF}" \
	     --point "dot product 32M"    "$${DOT_AI}"     "$${DOT_GF}" \
	     --point "saxpy 32M"          "$${SAXPY_AI}"   "$${SAXPY_GF}" \
	     --point "7-pt stencil 256³" "$${STEN_AI}"    "$${STEN_GF}" \
	     --point "SpMV 4M×7"        "$${SPMV_AI}"    "$${SPMV_GF}" \
	     --point "N-body 4096"        "$${NBODY_AI}"   "$${NBODY_GF}" \
	     --point "naive mm scalar"        "$${MM_AI}" "$${MM_SC}" \
	     --point "naive mm scalar+FMA"    "$${MM_AI}" "$${MM_SCF}" \
	     --point "naive mm SIMD"          "$${MM_AI}" "$${MM_SI}" \
	     --point "naive mm SIMD+FMA"      "$${MM_AI}" "$${MM_SIF}"

# Multicore version: same benchmarks, all cores. stream_mt/peak_flops_mt
# give the aggregate (shared-bandwidth / N-core-compute) ceilings; each
# kernel is measured with its work partitioned across std::thread::
# hardware_concurrency() threads. matmul_bench/openblas_bench are excluded
# here for the same reason as the single-thread plot (AMX, not these
# ceilings) — and Accelerate/OpenBLAS already have their own internal
# threading, which isn't this suite's to control.
plot_mt: all_mt
	@echo "=== Memory Bandwidth (STREAM, aggregate) ==="
	@./stream_mt | tee /tmp/stream_mt_out.txt
	@echo ""
	@echo "=== Peak FLOPS (aggregate) ==="
	@./peak_flops_mt | tee /tmp/flops_mt_out.txt
	@echo ""
	@echo "=== std::sort 16M floats (MT) ==="
	@./sort_bench_mt | tee /tmp/sort_mt_out.txt
	@echo ""
	@echo "=== dot product 32M floats (MT) ==="
	@./dot_bench_mt | tee /tmp/dot_mt_out.txt
	@echo ""
	@echo "=== saxpy 32M floats (MT) ==="
	@./saxpy_bench_mt | tee /tmp/saxpy_mt_out.txt
	@echo ""
	@echo "=== 7-point stencil 256^3 (MT) ==="
	@./stencil_bench_mt | tee /tmp/stencil_mt_out.txt
	@echo ""
	@echo "=== SpMV 4M rows x 7 nnz/row (MT) ==="
	@./spmv_bench_mt | tee /tmp/spmv_mt_out.txt
	@echo ""
	@echo "=== N-body 4096 particles (MT) ==="
	@./nbody_bench_mt | tee /tmp/nbody_mt_out.txt
	@echo ""
	@echo "=== Naive sgemm 512x512, no BLAS/AMX (MT) ==="
	@./matmul_naive_bench_mt | tee /tmp/matmul_naive_mt_out.txt
	@THREADS=$$(grep "^MT_THREADS=" /tmp/stream_mt_out.txt | cut -d= -f2); \
	 BW=$$(grep      "^PEAK_BANDWIDTH_MT="            /tmp/stream_mt_out.txt | cut -d= -f2); \
	 GF_SFMA=$$(grep "^PEAK_GFLOPS_SIMD_FMA_MT="     /tmp/flops_mt_out.txt  | cut -d= -f2); \
	 GF_SMUL=$$(grep "^PEAK_GFLOPS_SIMD_NOFMA_MT="   /tmp/flops_mt_out.txt  | cut -d= -f2); \
	 GF_SCFMA=$$(grep "^PEAK_GFLOPS_SCALAR_FMA_MT="   /tmp/flops_mt_out.txt  | cut -d= -f2); \
	 GF_SCMUL=$$(grep "^PEAK_GFLOPS_SCALAR_NOFMA_MT=" /tmp/flops_mt_out.txt  | cut -d= -f2); \
	 SORT_AI=$$(grep  "^SORT_AI_MT="                   /tmp/sort_mt_out.txt   | cut -d= -f2); \
	 SORT_GF=$$(grep  "^SORT_GFLOPS_MT="               /tmp/sort_mt_out.txt   | cut -d= -f2); \
	 DOT_AI=$$(grep   "^DOTPROD_AI_MT="                /tmp/dot_mt_out.txt    | cut -d= -f2); \
	 DOT_GF=$$(grep   "^DOTPROD_GFLOPS_MT="            /tmp/dot_mt_out.txt    | cut -d= -f2); \
	 SAXPY_AI=$$(grep "^SAXPY_AI_MT="                  /tmp/saxpy_mt_out.txt  | cut -d= -f2); \
	 SAXPY_GF=$$(grep "^SAXPY_GFLOPS_MT="              /tmp/saxpy_mt_out.txt  | cut -d= -f2); \
	 STEN_AI=$$(grep  "^STENCIL_AI_MT="                /tmp/stencil_mt_out.txt| cut -d= -f2); \
	 STEN_GF=$$(grep  "^STENCIL_GFLOPS_MT="            /tmp/stencil_mt_out.txt| cut -d= -f2); \
	 SPMV_AI=$$(grep  "^SPMV_AI_MT="                   /tmp/spmv_mt_out.txt   | cut -d= -f2); \
	 SPMV_GF=$$(grep  "^SPMV_GFLOPS_MT="               /tmp/spmv_mt_out.txt   | cut -d= -f2); \
	 NBODY_AI=$$(grep "^NBODY_AI_MT="                  /tmp/nbody_mt_out.txt  | cut -d= -f2); \
	 NBODY_GF=$$(grep "^NBODY_GFLOPS_MT="              /tmp/nbody_mt_out.txt  | cut -d= -f2); \
	 MM_AI=$$(grep    "^MATMUL_AI_MT="                 /tmp/matmul_naive_mt_out.txt | cut -d= -f2); \
	 MM_SC=$$(grep    "^MATMUL_SCALAR_MT_GFLOPS="      /tmp/matmul_naive_mt_out.txt | cut -d= -f2); \
	 MM_SCF=$$(grep   "^MATMUL_SCALAR_FMA_MT_GFLOPS="  /tmp/matmul_naive_mt_out.txt | cut -d= -f2); \
	 MM_SI=$$(grep    "^MATMUL_SIMD_MT_GFLOPS="        /tmp/matmul_naive_mt_out.txt | cut -d= -f2); \
	 MM_SIF=$$(grep   "^MATMUL_SIMD_FMA_MT_GFLOPS="    /tmp/matmul_naive_mt_out.txt | cut -d= -f2); \
	 python3 roofline_multicore.py "$${BW}" "$${GF_SFMA}" "$${GF_SMUL}" "$${GF_SCFMA}" "$${GF_SCMUL}" --threads "$${THREADS}" --machine "$(MACHINE)" --cores "$(CORES)" \
	     --point "std::sort 16M"      "$${SORT_AI}"    "$${SORT_GF}" \
	     --point "dot product 32M"    "$${DOT_AI}"     "$${DOT_GF}" \
	     --point "saxpy 32M"          "$${SAXPY_AI}"   "$${SAXPY_GF}" \
	     --point "7-pt stencil 256³" "$${STEN_AI}"    "$${STEN_GF}" \
	     --point "SpMV 4M×7"        "$${SPMV_AI}"    "$${SPMV_GF}" \
	     --point "N-body 4096"        "$${NBODY_AI}"   "$${NBODY_GF}" \
	     --point "naive mm scalar"        "$${MM_AI}" "$${MM_SC}" \
	     --point "naive mm scalar+FMA"    "$${MM_AI}" "$${MM_SCF}" \
	     --point "naive mm SIMD"          "$${MM_AI}" "$${MM_SI}" \
	     --point "naive mm SIMD+FMA"      "$${MM_AI}" "$${MM_SIF}"

clean:
	rm -f stream peak_flops sort_bench dot_bench saxpy_bench stencil_bench spmv_bench nbody_bench \
	      matmul_naive_bench matmul_bench openblas_bench \
	      stream_mt peak_flops_mt sort_bench_mt dot_bench_mt saxpy_bench_mt stencil_bench_mt \
	      spmv_bench_mt nbody_bench_mt matmul_naive_bench_mt \
	      *.o roofline.png roofline_multicore.png \
	      /tmp/stream_out.txt /tmp/flops_out.txt /tmp/sort_out.txt /tmp/dot_out.txt \
	      /tmp/saxpy_out.txt /tmp/stencil_out.txt /tmp/spmv_out.txt /tmp/nbody_out.txt \
	      /tmp/matmul_naive_out.txt \
	      /tmp/mm1024_out.txt /tmp/mm4096_out.txt /tmp/ob1024_out.txt /tmp/ob4096_out.txt \
	      /tmp/stream_mt_out.txt /tmp/flops_mt_out.txt /tmp/sort_mt_out.txt /tmp/dot_mt_out.txt \
	      /tmp/saxpy_mt_out.txt /tmp/stencil_mt_out.txt /tmp/spmv_mt_out.txt /tmp/nbody_mt_out.txt \
	      /tmp/matmul_naive_mt_out.txt
