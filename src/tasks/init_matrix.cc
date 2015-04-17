#include "init_matrix.hpp"
#include "ptr_matrix.hpp"

#include "utility.hpp" // for FIELDID_V
#include <assert.h>

int InitMatrixTask::TASKID;

InitMatrixTask::InitMatrixTask(Domain domain,
			       TaskArgument global_arg,
			       ArgumentMap arg_map,
			       Predicate pred,
			       bool must,
			       MapperID id,
			       MappingTagID tag)
  
  : IndexLauncher(TASKID, domain, global_arg,
		  arg_map, pred, must, id, tag) {}

void InitMatrixTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <InitMatrixTask::cpu_task>(AUTO_GENERATE_ID,
			       Processor::LOC_PROC, 
			       false,
			       true,
			       AUTO_GENERATE_ID,
			       TaskConfigOptions(true/*leaf*/),
			       "Init_Matrix");

  //#ifndef SHOW_REGISTER_TASKS
  printf("Register task %d : Init_Matrix\n", TASKID);
  //#endif
}

void InitMatrixTask::cpu_task(const Task *task,
			      const std::vector<PhysicalRegion> &regions,
			      Context ctx, HighLevelRuntime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->arglen == sizeof(TaskArgs));
  assert(task->local_arglen == sizeof(long));

  Point<1> p = task->index_point.get_point<1>();
  printf("point = %d\n", p[0]);

  const long seed = *((const long*)task->local_args);
  printf("random seed = %lu \n", seed);
  
  const TaskArgs blockSize = *((const TaskArgs*)task->args);
  int rblk  = blockSize.rblk;
  int cblk  = blockSize.cblk;
  int clo   = blockSize.clo;
  int chi   = blockSize.chi;
  //printf("block row size = %i\n", rows);
  //printf("block col size = %i\n", cols);

  int rlo = p[0]*rblk;
  int rhi = (p[0] + 1) * rblk;

  /*
  double *base = region_pointer(regions[0], rlo, rhi, clo, chi);
  int colIdx = 0;
  while (colIdx < chi-clo) {
    double *ptr = base + colIdx*rblk;
    PtrMatrix pMat(rblk, cblk, rblk, ptr);
    pMat.rand(seed);
    colIdx += cblk;
  }
  */

  while (clo+cblk <= chi) {
    PtrMatrix A = get_raw_pointer(regions[0], rlo, rhi, clo, clo+cblk);
    A.rand(seed);
    clo += cblk;
  }  
}



