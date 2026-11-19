# Matrix multiply with Arm SME, on the Apple matrix unit. M4 or later only.
#
#   make           build mm_sme
#   make run       peak rate, then every kernel at n = 1024 and 2048
#                  (writes results_sme.M5.csv; about 15 s)

CXX     = clang++
RESULTS = results_sme.M5.csv
HAVE_SME := $(shell sysctl -n hw.optional.arm.FEAT_SME 2>/dev/null)

KERNELS = transpose sme1 sme4 sme4_pack_k256 sme4_pack_k512 sme4_pack_k1024 accelerate

all: mm_sme

# +sme is needed even with -march=native: Apple clang does not turn it on.
mm_sme: mm_sme.cpp
ifneq ($(HAVE_SME),1)
	@echo "This machine has no SME (needs Apple silicon M4 or later)."; exit 1
endif
	$(CXX) -std=c++17 -O3 -march=native+sme -o $@ $< -framework Accelerate

# One thread for Accelerate, so it is compared with our one-thread kernels.
$(RESULTS): mm_sme
	rm -f $@
	./mm_sme peak | tee -a $@
	for n in 1024 2048; do VECLIB_MAXIMUM_THREADS=1 ./mm_sme $$n $(KERNELS) | tee -a $@; done

run: $(RESULTS)

clean:
	rm -f mm_sme

.PHONY: all run clean
