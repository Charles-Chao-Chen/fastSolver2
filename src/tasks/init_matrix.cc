#include "init_matrix.hpp"
#include "ptr_matrix.hpp"

#include "utility.hpp" // for FIELDID_V
#include <assert.h>

static Realm::Logger log_solver_tasks("solver_tasks");

int InitMatrixTask::TASKID;

InitMatrixTask::InitMatrixTask(Domain domain,
			       TaskArgument global_arg,
			       ArgumentMap arg_map,
			       MappingTagID tag,
			       Predicate pred,
			       bool must,
			       MapperID id)
  
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

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : Init_Matrix\n", TASKID);
#endif
}

void InitMatrixTask::cpu_task(const Task *task,
			      const std::vector<PhysicalRegion> &regions,
			      Context ctx, HighLevelRuntime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->arglen == sizeof(TaskArgs));
  //assert(task->local_arglen == sizeof(long));
  log_solver_tasks.print("Inside init tasks.");

  Point<1> p = task->index_point.get_point<1>();
  //printf("point = %d\n", p[0]);

  const long nPart = *((const long*)task->local_args);
  //printf("nPart = %lu \n", nPart);

  const TaskArgs blockSize = *((const TaskArgs*)task->args);
  int rblk  = blockSize.rblk;
  int cblk  = blockSize.cblk;
  int chi   = blockSize.chi;
  //printf("block row size = %i\n", rows);
  //printf("block col size = %i\n", cols);

  int rlo = p[0]*rblk;
  //int rhi = (p[0] + 1) * rblk;

  int blksmall = rblk / nPart;
  for (int i=0; i<nPart; i++) {
    const long seed = *((const long*)task->local_args + i + 1);
    //printf(" seed = %lu \n", seed);
    int clo   = blockSize.clo;
    while (clo+cblk <= chi) {
      PtrMatrix A = get_raw_pointer(regions[0], rlo+i*blksmall, rlo+(i+1)*blksmall, clo, clo+cblk);
      A.rand(seed);
      //A.display("sub-mat");
      //std::cout<<"LD:"<<A.LD()<<std::endl;
      clo += cblk;
    }
  }
}
