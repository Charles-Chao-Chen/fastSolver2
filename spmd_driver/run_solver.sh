#!/bin/bash

#Nodes="n0000,n0001,n0002,n0003"
#NumNodes=4
Nodes="n0000"
NumNodes=1
NumCores=2

TreeLevel=8
Ofile=$NumNodes"node_%.log"

time mpirun -H $Nodes \
	-bind-to none \
	-x GASNET_BACKTRACE=1 -x GASNET_IB_SPAWNER=mpi \
	-x LEGION_FREEZE_ON_ERROR=1 \
	./solver \
	-machine $NumNodes \
	-core $NumCores \
	-mtxlvl $TreeLevel \
	-ll:cpu $NumCores \
	-ll:csize 2000 \
    	-hl:prof 4 \
	-logfile $Ofile


