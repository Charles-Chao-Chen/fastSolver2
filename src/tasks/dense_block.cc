#include "dense_block.hpp"
#include "ptr_matrix.hpp"

#include "utility.hpp" // for FIELDID_V
#include <assert.h>

static Realm::Logger log_solver_tasks("solver_tasks");

int DenseBlockTask::TASKID;

DenseBlockTask::DenseBlockTask(Domain domain,
			       TaskArgument global_arg,
			       ArgumentMap arg_map,
			       MappingTagID tag,
			       Predicate pred,
			       bool must,
			       MapperID id)
  
  : IndexLauncher(TASKID, domain, global_arg,
		  arg_map, pred, must, id, tag) {}

void DenseBlockTask::register_tasks(void)
{
  TASKID = Runtime::register_legion_task
    <DenseBlockTask::cpu_task>(AUTO_GENERATE_ID,
			       Processor::LOC_PROC, 
			       false,
			       true,
			       AUTO_GENERATE_ID,
			       TaskConfigOptions(true/*leaf*/),
			       "Dense_Block");

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : Dense_Block\n", TASKID);
#endif
}

void DenseBlockTask::cpu_task(const Task *task,
			      const std::vector<PhysicalRegion> &regions,
			      Context ctx, Runtime *runtime) {

  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->arglen == sizeof(TaskArgs));
  //  assert(task->local_arglen == sizeof(ThreeSeeds));
  log_solver_tasks.print("Inside init dense block tasks.");
  
  Point<1> p(task->index_point);
  //printf("point = %d\n", p[0]);

  //const ThreeSeeds seeds = *((const ThreeSeeds*)task->local_args);
  //long uSeed = seeds.uSeed;
  //long vSeed = seeds.vSeed;
  //long dSeed = seeds.dSeed;
  //printf("random seeds = (%lu, %lu, %lu) \n", uSeed, vSeed, dSeed);
  
  const TaskArgs matrix = *((const TaskArgs*)task->args);
  int nrow = matrix.size;
  int rank = matrix.rank;
  int ofst = matrix.offset;
  int rlo = p[0]*nrow;
  //  int rhi = (p[0]+1)*nrow;
  
  const long nPart = *((const long*)task->local_args);
  int rblk = nrow / nPart;
  for (int i=0; i<nPart; i++) {
    PtrMatrix K = get_raw_pointer(regions[0], rlo+i*rblk, rlo+(i+1)*rblk, 0, rblk);
    // recover U, V and D
    PtrMatrix U(rblk, rank), V(rblk, rank), D(rblk, 1);
    const long uSeed = *((const long*)task->local_args + 1 + 3*i + 0);
    const long vSeed = *((const long*)task->local_args + 1 + 3*i + 1);
    const long dSeed = *((const long*)task->local_args + 1 + 3*i + 2);
    U.rand(uSeed);
    V.rand(vSeed);
    D.rand(dSeed, ofst);
    V.set_trans('t');
    PtrMatrix::gemm(U, V, D, K);
  }
}
