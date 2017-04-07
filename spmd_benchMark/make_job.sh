#!/bin/sh

NumNodes=8 #512
NumCore=12
Mtxlvl=9 #17
Time=00:10:00

echo Run legion solver on `echo $NumNodes` nodes...
sbatch --nodes=${NumNodes} \
	--cpus-per-task=${NumCore} \
	--time=${Time} job_solver.slm \
	${Mtxlvl}


