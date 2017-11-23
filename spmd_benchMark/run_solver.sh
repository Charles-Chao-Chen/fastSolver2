#!/bin/bash

Nn=2
TreeLevel=4

Ofile=$Nn"node_%.log"

#	-x GASNET_IB_SPAWNER=mpi  \
#	-level inst=2,metadata=2 \

#mpirun -np $Nn -pernode \
mpirun -np $Nn -npersocket 1 \
	-bind-to socket --report-bindings \
	-x GASNET_BACKTRACE=1     \
	-x LEGION_FREEZE_ON_ERROR=1 \
	./solver \
	-machine $Nn \
	-core 2 \
	-mtxlvl $TreeLevel \
	-rank0 10 \
	-rank1 100 \
	-ll:cpu 1 \
	-hl:prof 2 \
	-level 4,legion_prof=2 \
	-logfile $Ofile &


#	-ll:csize 32000 \
#	-hl:sched 1024 \
#	-bind-to none --report-bindings \


