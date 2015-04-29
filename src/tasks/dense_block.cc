#include "dense_block.hpp"
#include "ptr_matrix.hpp"

#include "utility.hpp" // for FIELDID_V
#include <assert.h>

int DenseBlockTask::TASKID;

DenseBlockTask::DenseBlockTask(Domain domain,
			       TaskArgument global_arg,
			       ArgumentMap arg_map,
			       Predicate pred,
			       bool must,
			       MapperID id,
			       MappingTagID tag)
  
  : IndexLauncher(TASKID, domain, global_arg,
		  arg_map, pred, must, id, tag) {}

void DenseBlockTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <DenseBlockTask::cpu_task>(AUTO_GENERATE_ID,
			       Processor::LOC_PROC, 
			       false,
			       true,
			       AUTO_GENERATE_ID,
			       TaskConfigOptions(true/*leaf*/),
			       "Dense_Block");

  //#ifndef SHOW_REGISTER_TASKS
  printf("Register task %d : Dense_Block\n", TASKID);
  //#endif
}

void DenseBlockTask::cpu_task(const Task *task,
			      const std::vector<PhysicalRegion> &regions,
			      Context ctx, HighLevelRuntime *runtime) {

  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->arglen == sizeof(TaskArgs));
  assert(task->local_arglen == sizeof(ThreeSeeds));
  Point<1> p = task->index_point.get_point<1>();
  //printf("point = %d\n", p[0]);

  const ThreeSeeds seeds = *((const ThreeSeeds*)task->local_args);
  long uSeed = seeds.uSeed;
  long vSeed = seeds.vSeed;
  long dSeed = seeds.dSeed;
  //  printf("random seeds = (%lu, %lu, %lu) \n", uSeed, vSeed, dSeed);
  
  const TaskArgs matrix = *((const TaskArgs*)task->args);
  int size = matrix.size;
  int rank = matrix.rank;
  int ofst = matrix.offset;
  int rlo = p[0]*size;
  int rhi = (p[0]+1)*size;
  
  PtrMatrix K = get_raw_pointer(regions[0], rlo, rhi, 0, size);

  // recover U, V and D
  PtrMatrix U(size, rank), V(size, rank), D(size, 1);
  U.rand(uSeed);
  V.rand(vSeed);
  D.rand(dSeed, ofst);
  V.set_trans('t');
  PtrMatrix::gemm(U, V, D, K);
  
  /*
  double *base = region_pointer(regions[0], rlo, rhi, 0, cols);

  // recover U, V and D
  PtrMatrix U(rows, rank), V(rows, rank), D(rows, 1);
  U.rand(uSeed);
  V.rand(vSeed);
  D.rand(dSeed);

  int bSize = rows / blks;
  assert( bSize == cols );
  for (int i=0; i<blks; i++) {
    char trans = 't';
    PtrMatrix Ublk(bSize, rank, rows, U.pointer()+i*bSize);
    PtrMatrix Vblk(bSize, rank, rows, V.pointer()+i*bSize, trans);
    PtrMatrix Dblk(bSize, 1,    rows, D.pointer()+i*bSize);
    PtrMatrix pMat(bSize, cols, rows, base       +i*bSize);
    PtrMatrix::gemm(Ublk, Vblk, Dblk, pMat);
  }
  */
}
