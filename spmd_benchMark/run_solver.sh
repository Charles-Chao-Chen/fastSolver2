#!/bin/bash

Nn=1
Ofile=$Nn"node_%.log"

#	-x GASNET_IB_SPAWNER=mpi  \
#	-level inst=2,metadata=2 \

mpirun -np $Nn -pernode \
	-bind-to none -report-bindings \
	-x GASNET_BACKTRACE=1     \
	-x LEGION_FREEZE_ON_ERROR=1 \
	./solver \
	-machine $Nn \
	-core 8 \
	-mtxlvl 6 \
	-ll:cpu 8 \
	-hl:prof $Nn \
	-level 4,legion_prof=2 \
	-logfile $Ofile
