ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

#Flags for directing the runtime makefile what to include
DEBUG ?= 0                   # Include debugging symbols
OUTPUT_LEVEL ?= LEVEL_DEBUG  # Compile time print level
SHARED_LOWLEVEL ?= 0	  # Use the shared low level
USE_CUDA ?= 0
USE_GASNET ?= 1
USE_HDF ?= 0
#ALT_MAPPERS ?= 1		  # Compile the alternative mappers

# Put the binary file name here
OUTFILE	:= 	solver

# List all the application source files here
GEN_SRC	:= 	solver.cc \
		../src/hmatrix.cc ../src/tree.cc \
		../src/lmatrix.cc ../src/matrix.cc \
		../src/tasks/leaf_solve.cc ../src/tasks/node_solve.cc \
		../src/tasks/gemm_reduce.cc ../src/tasks/gemm_broadcast.cc \
		../src/tasks/gemm.cc ../src/tasks/gemm_inplace.cc \
		../src/tasks/node_solve_region.cc \
		../src/tasks/projector.cc ../src/tasks/reduce_add.cc \
		../src/tasks/init_matrix.cc ../src/tasks/clear_matrix.cc \
		../src/tasks/solver_tasks.cc ../src/tasks/display_matrix.cc \
		../src/tasks/dense_block.cc ../src/tasks/add_matrix.cc \
		../src/ptr_matrix.cc ../src/utility.cc \
		../src/tasks/scale_matrix.cc \
		../src/tasks/new_mapper.cc
#		../src/tasks/mapper.cc \
		../src/tasks/dist_mapper.cc ../src/tasks/spmd_mapper.cc

GEN_GPU_SRC	:=				# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?= -I ../include/ -I ../include/tasks/
CC_FLAGS	?=
NVCC_FLAGS	?=
GASNET_FLAGS	?=

ifneq (${CRAYPE_VERSION},)
# Cray systems include a special version of BLAS and LAPACK which is
# automatically linked into every application. Unfortunately, these
# versions include automatic multi-threading which breaks the
# application. To avoid this breakage, we use a custom version of BLAS.
LD_FLAGS 	:= ../netlib_blas/dgemm.o ../netlib_blas/dgesv.o ../netlib_blas/dgetrf.o ../netlib_blas/dgetrf2.o ../netlib_blas/dgetrs.o ../netlib_blas/dlamch.o ../netlib_blas/dlaswp.o ../netlib_blas/dscal.o ../netlib_blas/dtrsm.o ../netlib_blas/idamax.o ../netlib_blas/ieeeck.o ../netlib_blas/ilaenv.o ../netlib_blas/iparmq.o ../netlib_blas/lsame.o ../netlib_blas/xerbla.o
else
# otherwise use system BLAS/LAPACK
LD_FLAGS 	:= -llapack -lblas
endif

# enable legion spy
# ifeq ($(strip $(DEBUG)),1)
# CC_FLAGS += -DLEGION_SPY \
# 	-DPRIVILEGE_CHECKS \
# 	-DBOUNDS_CHECKS
# endif

# dynamic linking on daint
ifeq ($(findstring daint,$(shell uname -n)),daint)
LD_FLAGS += -dynamic
endif

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

include $(LG_RT_DIR)/runtime.mk


#clean solver related
cleansolver:
	@$(RM) -rf $(OUTFILE) $(GEN_OBJS)  \
	*~  \
	../include/*~ ../src/*~ \
	../include/tasks/*~ ../src/tasks/*~

tar:	
	tar cvfz fastSolver.tgz Makefile \
	solver.cc \
	../src/hmatrix.cc ../src/tree.cc \
	../src/lmatrix.cc ../src/matrix.cc \
	../src/tasks/leaf_solve.cc ../src/tasks/node_solve.cc \
	../src/tasks/gemm_reduce.cc   ../src/tasks/gemm_broadcast.cc \
	../src/tasks/projector.cc ../src/tasks/reduce_add.cc \
	../src/tasks/init_matrix.cc ../src/tasks/clear_matrix.cc \
	../src/tasks/solver_tasks.cc ../src/tasks/display_matrix.cc \
	../src/tasks/dense_block.cc ../src/tasks/add_matrix.cc \
	../src/ptr_matrix.cc ../src/utility.cc \
	../src/tasks/scale_matrix.cc ../src/tasks/mapper.cc \
	../src/tasks/dist_mapper.cc \
	\
	../include/hmatrix.hpp ../include/tree.hpp \
	../include/lmatrix.hpp ../include/matrix.hpp \
	../include/tasks/leaf_solve.hpp ../include/tasks/node_solve.hpp \
	../include/tasks/gemm_reduce.hpp   ../include/tasks/gemm_broadcast.hpp \
	../include/tasks/projector.hpp ../include/tasks/reduce_add.hpp \
	../include/tasks/init_matrix.hpp ../include/tasks/clear_matrix.hpp \
	../include/tasks/solver_tasks.hpp ../include/tasks/display_matrix.hpp \
	../include/tasks/dense_block.hpp ../include/tasks/add_matrix.hpp \
	../include/ptr_matrix.hpp ../include/utility.hpp \
	../include/lapack_blas.hpp \
	../include/tasks/scale_matrix.hpp ../include/tasks/mapper.hpp \
	../include/tasks/dist_mapper.hpp
