#!/bin/sh

# 2, 3, 5 may be a good small test case for scalability
# while a large test case is 64, 8(2+6), 13 

NumNodes=64
Launchlvl=8
Treelvl=13
Rank=100
Niter=3
Tracing=0
#Time=00:05:00

export NumProcs
echo Run legion solver at `echo $NumNodes` nodes...
sbatch --nodes=${NumNodes} --ntasks=${NumNodes} \
	--time=00:03:00 job_solver.slm \
	${NumNodes} ${Launchlvl} ${Treelvl} ${Rank} ${Niter} \
	${Tracing}
