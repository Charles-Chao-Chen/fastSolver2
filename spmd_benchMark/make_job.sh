#!/bin/sh

NumNodes=16
NumCore=4
Mtxlvl=12
#Time=00:05:00

export NumProcs
echo Run legion solver at `echo $NumNodes` nodes...
sbatch --nodes=${NumNodes} \
	--ntasks=${NumNodes} \
	--time=00:05:00 job_solver.slm \
	${NumNodes} ${NumCore} ${Mtxlvl}
