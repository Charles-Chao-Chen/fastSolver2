#!/bin/sh

NumNodes=4
NumRanksPerNode=8
#NumCoresPerRank=1
NumTasksPerRank=8
Mtxlvl=11
Time=00:30:00

let "NumRanks = $NumNodes*$NumRanksPerNode"

echo Run legion solver on `echo $NumNodes` nodes...
sbatch --nodes=${NumNodes} \
	--ntasks-per-node=${NumRanksPerNode} \
	--time=${Time} \
	--output=${NumRanks}rank_ref${NumTasksPerRank}_job.out \
	--error=${NumRanks}rank_ref${NumTasksPerRank}_job.err \
	job_solver.slm \
	${Mtxlvl} \
	${NumTasksPerRank}

#	--cpus-per-task=${NumCoresPerRank} \

