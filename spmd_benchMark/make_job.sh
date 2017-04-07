#!/bin/sh

NumNodes=4 #512
NumCore=12
Mtxlvl=8 #17
Time=00:05:00

echo Run legion solver on `echo $NumNodes` nodes...
sbatch --nodes=${NumNodes} \
	--cpus-per-task=${NumCore} \
	--time=${Time} job_solver.slm \
	${Mtxlvl}


