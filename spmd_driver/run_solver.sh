#!/bin/bash

#Nodes="n0000,n0001,n0002,n0003"
#NumNodes=4
#Nodes="n0000"
NumNodes=2
NumCores=8
NumThrds=$NumNodes*$NumCores

echo $NumThrds

TreeLevel=4
Ofile=$NumNodes"node_%.log"

time mpirun -np $NumNodes \
	-bind-to none --report-bindings \
	-x GASNET_BACKTRACE=1 -x GASNET_IB_SPAWNER=mpi \
	-x LEGION_FREEZE_ON_ERROR=1 \
	./solver \
	-machine $NumThrds \
	-core 1 \
	-mtxlvl $TreeLevel \
	-ll:cpu $NumCores \
    	-hl:prof $NumNodes \
	-logfile $Ofile

#	-ll:csize 2000 \
#	-level 4,legion_prof=2 \

