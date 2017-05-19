#!/bin/sh

NumNodes=1
Mtxlvl=11
NumCore=12
Time=00:10:00

echo Run legion solver on `echo $NumNodes` nodes...
sbatch --nodes=${NumNodes} \
	--cpus-per-task=${NumCore} \
	--time=${Time} \
	--output=${NumNodes}node_job.out \
	--error=${NumNodes}node_job.err \
	job_solver.slm \
	${Mtxlvl}


