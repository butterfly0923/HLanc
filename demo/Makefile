NVCC       = nvcc
NVCC_ARCH  = sm_20
NVCC_FLAGS = -O2 -Xptxas --verbose -arch $(NVCC_ARCH)
NVCC_LIBS  = -lcublas -lcusparse
MKL_PATH   = /opt/intel/mkl
MKL_LIBS   = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
SHELL      = /bin/bash
PROG       = HLanc_demo HLanc_solver verify_eigs

default: $(PROG)
	. $(MKL_PATH)/bin/mklvars.sh intel64;	\
	./HLanc_demo

%: %.cu
	. $(MKL_PATH)/bin/mklvars.sh intel64;	\
	$(NVCC) -o $@ $(NVCC_FLAGS) $(NVCC_LIBS) $(MKL_LIBS) $<

clean:
	rm -rf $(PROG) simple.bin

.PHONY: run clean
