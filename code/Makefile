TARGET = tc_cuda
TARGET_HASHJOIN = hashjoin
COMMON = common
NVCC_FLAGS += -dlto -lineinfo
LDFLAGS = -lmpi -lnccl
COMPILER_FLAGS = -O3 -w
DEBUG_FLAGS = -DDEBUG
THETA_FLAGS = -L/lus/theta-fs0/software/thetagpu/openmpi/openmpi-4.1.4_ucx-1.12.1_gcc-9.4.0/lib
SRC = $(TARGET).cu
SRC_HASHJOIN = $(TARGET_HASHJOIN).cu

build:
	nvcc $(SRC) -o $(TARGET).out $(COMPILER_FLAGS)

test:
	./$(TARGET).out

run: build test

debug:
	nvcc $(SRC) -o $(TARGET).out $(COMPILER_FLAGS) $(DEBUG_FLAGS)
	./$(TARGET).out

hashjoin:
	nvcc $(SRC_HASHJOIN) -o $(TARGET_HASHJOIN).out $(COMPILER_FLAGS)
	./$(TARGET_HASHJOIN).out

clean:
	rm -f $(TARGET).out

.PHONY.: build