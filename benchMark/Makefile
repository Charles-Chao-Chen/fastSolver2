
ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

#Flags for directing the runtime makefile what to include
DEBUG=0                   # Include debugging symbols
OUTPUT_LEVEL=LEVEL_DEBUG  # Compile time print level
SHARED_LOWLEVEL=0	  # Use the shared low level
USE_CUDA=0
#ALT_MAPPERS=1		  # Compile the alternative mappers

# Put the binary file name here
OUTFILE	:= 	solver

# List all the application source files here
GEN_SRC	:= 	solver.cc

GEN_GPU_SRC	:=				# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	:= 	
CC_FLAGS	:=

#-g  \
			$(INC_FLAGS)  \
			-DLEGION_PROF  \
			-DLEGION_SPY   \
			-DNODE_LOGGING \
#			-DSERIAL  \
#			-DDEBUG
#			-DLEGION_BACKTRACE  \

#		   -DSHOW_REGISTER_TASKS \
#		   -g		\

#		   -DNDEBUG	  \
#		   -DDEBUG_GEMM
#		   -DDEBUG_NODE_SOLVE

NVCC_FLAGS	:=
GASNET_FLAGS	:=

# gnu blas and lapack
LD_FLAGS	:= -L /usr/lib/	-llapack -lblas -lm

# mkl linking flags
#LD_FLAGS := -L/share/apps/intel/intel-14/mkl/lib/intel64/ \
	-L/share/apps/intel/intel-14/lib/intel64/ \
	-lmkl_intel_lp64 	\
	-lmkl_core		\
	-lmkl_sequential	\
	-lpthread 		\
	-lm

#	-lmkl_intel_thread 	\
	-liomp5 		\
	-lmkl_sequential	\



###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

# All these variables will be filled in by the runtime makefile
LOW_RUNTIME_SRC	:=
HIGH_RUNTIME_SRC:=
GPU_RUNTIME_SRC	:=
MAPPER_SRC	:=

include $(LG_RT_DIR)/runtime.mk

# General shell commands
SHELL	:= /bin/sh
SH	:= sh
RM	:= rm -f
LS	:= ls
MKDIR	:= mkdir
MV	:= mv
CP	:= cp
SED	:= sed
ECHO	:= echo
TOUCH	:= touch
MAKE	:= make
ifndef GCC
GCC	:= g++
endif
ifndef NVCC
NVCC	:= $(CUDA)/bin/nvcc
endif
SSH	:= ssh
SCP	:= scp

common_all : all

.PHONY	: common_all

GEN_OBJS	:= $(GEN_SRC:.cc=.o)
LOW_RUNTIME_OBJS:= $(LOW_RUNTIME_SRC:.cc=.o)
HIGH_RUNTIME_OBJS:=$(HIGH_RUNTIME_SRC:.cc=.o)
MAPPER_OBJS	:= $(MAPPER_SRC:.cc=.o)
# Only compile the gpu objects if we need to 
ifndef SHARED_LOWLEVEL
GEN_GPU_OBJS	:= $(GEN_GPU_SRC:.cu=.o)
GPU_RUNTIME_OBJS:= $(GPU_RUNTIME_SRC:.cu=.o)
else
GEN_GPU_OBJS	:=
GPU_RUNTIME_OBJS:=
endif

ALL_OBJS	:= $(GEN_OBJS) $(GEN_GPU_OBJS) $(LOW_RUNTIME_OBJS) $(HIGH_RUNTIME_OBJS) $(GPU_RUNTIME_OBJS) $(MAPPER_OBJS)

all:
	$(MAKE) $(OUTFILE)

# If we're using the general low-level runtime we have to link with nvcc
$(OUTFILE) : $(ALL_OBJS)
	@echo "*********** Linking GNU ***********"
	@echo "---> Linking objects into one binary: $(OUTFILE)"
ifdef SHARED_LOWLEVEL
	$(GCC) -o $(OUTFILE) $(ALL_OBJS) $(LD_FLAGS) $(GASNET_FLAGS)
else
	$(NVCC) -o $(OUTFILE) $(ALL_OBJS) $(LD_FLAGS) $(GASNET_FLAGS)
endif

$(GEN_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(LOW_RUNTIME_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(HIGH_RUNTIME_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(MAPPER_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(GEN_GPU_OBJS) : %.o : %.cu
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)

$(GPU_RUNTIME_OBJS): %.o : %.cu
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)

clean:
	@$(RM) -rf $(OUTFILE) $(GEN_OBJS)  \
	*~  \
	../include/*~ ../src/*~ \

cleanall:
	@$(RM) -rf $(ALL_OBJS) *~

tar:	
	tar cvfz fastSolver.tgz

r2n:
	mpiexec -n 2 -ppn 1	\
	-env MV2_SHOW_CPU_BINDING=1 \
	-env MV2_ENABLE_AFFINITY=0  \
	-env GASNET_IB_SPAWNER=mpi  \
	-env GASNET_BACKTRACE=1     \
	./main			\
	-np 2 			\
	-level 5 		\
	-ll:cpu    8		\
	-ll:util   4 		\
	-ll:csize  30000 	\
	-hl:sched  8192  	\
	-hl:window 8192

# --- legion profile ---

# nproc : the number of processes, and there should be one
#  process for every node
# numa  : can be set to 'numactl -m 0 -N 0' to use numa node
# ncpu  :
# nutil :
# leaf  : legion leaf size
prof:
	mpiexec -n $(nproc) -ppn 1  \
	-env MV2_SHOW_CPU_BINDING=1 \
	-env MV2_ENABLE_AFFINITY=0  \
	-env GASNET_IB_SPAWNER=mpi  \
	-env GASNET_BACKTRACE=1     \
	./multi_launch			\
	-test $(test)		\
	-np $(nproc) 		\
	-leaf $(leaf)		\
	-ll:cpu    $(ncpu)	\
	-ll:util   $(nutil)	\
	-ll:csize  20000 	\
	-hl:sched  8192  	\
	-hl:window 8192		\
	-level 2 		\
	-cat legion_prof	\

#	-hl:prof   1		\
#	numactl			\
	-m 0 -N 0		\
	$(numa) 		\


# --- legion spy ---
#spy2:\
	mpiexec -n 2 -ppn 1	\
	-env OMP_NUM_THREADS=1	\
	-env MV2_SHOW_CPU_BINDING=1 \
	-env MV2_ENABLE_AFFINITY=0  \
	-env GASNET_IB_SPAWNER=mpi  \
	-env GASNET_BACKTRACE=1     \
	./main -cat legion_spy -level 2 \
	-ll:cpu 12 -ll:csize 30000 \
	-hl:sched 8192 -hl:window 8192

# --- Sapling commands ---

# check task execution (data movement)
spy2:
	mpirun -H n0001,n0002 \
	-bind-to none \
	-x GASNET_IB_SPAWNER -x GASNET_BACKTRACE=1 \
	./main \
	-np 2 -leaf 1 \
	-cat legion_spy -level 2 \
	-ll:cpu 11 -ll:util 1 \
	-ll:csize 30000 \
	-hl:sched 8192
#	python ~/legion/tools/legion_spy.py -p node_0.log

prof2:
	mpirun -H n0001,n0002 \
	-bind-to none -report-bindings \
	-x GASNET_IB_SPAWNER -x GASNET_BACKTRACE=1 \
	./multi_launch \
	-leaf 1 \
	-ll:cpu 11 -ll:util 1 \
	-ll:csize 30000 \
	-hl:sched 8192 \
	-cat legion_prof -level 2 \
#	-cat legion_spy -level 2
#	-level 5

# --- legion profiler ---

# idx : node index
node:
	python ~/legion/tools/legion_prof.py -p node_$(idx).log
#	python ~/legion/tools/legion_prof.py -p node_$(idx).log \
	> legion_node.txt
