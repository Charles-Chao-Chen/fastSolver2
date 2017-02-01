#!/bin/sh

NumNodes=1 #512
NumCore=8
Mtxlvl=8 #17
Time=00:05:00

#export NumProcs
echo Run legion solver at `echo $NumNodes` nodes...
sbatch --nodes=${NumNodes} \
	--cpus-per-task=${NumCore} \
	--time=${Time} job_solver.slm \
	${Mtxlvl}


#	--ntasks=${NumNodes} \

