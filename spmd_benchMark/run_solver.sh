#!/bin/bash

Nn=128
TreeLevel=18

Ofile=$Nn"node_%.log"

#	-x GASNET_IB_SPAWNER=mpi  \
#	-level inst=2,metadata=2 \

mpirun -np $Nn -pernode \
	-bind-to none \
	-x GASNET_BACKTRACE=1     \
	-x LEGION_FREEZE_ON_ERROR=1 \
	./solver \
	-machine $Nn \
	-core 8 \
	-mtxlvl $TreeLevel \
	-ll:cpu 8 \
	-ll:csize 32000 \
	-hl:sched 1024 \
	-hl:prof 8 \
	-level 4,legion_prof=2 \
	-logfile $Ofile &
