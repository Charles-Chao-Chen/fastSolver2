#include "node_solve.hpp"
#include "ptr_matrix.hpp"
#include "utility.hpp"

int NodeSolveTask::TASKID;

NodeSolveTask::NodeSolveTask(Domain domain,
			     TaskArgument global_arg,
			     ArgumentMap arg_map,
			     MappingTagID tag)
  
  : IndexLauncher(TASKID, domain, global_arg, arg_map,
		  Predicate::TRUE_PRED, false, 0, tag) {}

void NodeSolveTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <NodeSolveTask::cpu_task>(AUTO_GENERATE_ID,
			      Processor::LOC_PROC, 
			      false,
			      true,
			      AUTO_GENERATE_ID,
			      TaskConfigOptions(true/*leaf*/),
			      "Node_Solve");

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : Node_Solve\n", TASKID);
#endif
}

// solve the following system for every partition
// --             --  --    --     --      --
// |  I     V1'*u1 |  | eta0 |     | V1'*d1 |
// |               |  |      |  =  |        |
// | V0'*u0   I    |  | eta1 |     | V0'*d0 |
// --             --  --    --     --      --
// note the reversed order in VTd
void NodeSolveTask::cpu_task(const Task *task,
			     const std::vector<PhysicalRegion> &regions,
			     Context ctx, HighLevelRuntime *runtime) {

  //std::cout<<"In node solve task."<<std::endl;
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(TaskArgs));
  Point<1> p = task->index_point.get_point<1>();
  //  printf("point = %d\n", p[0]);

  const TaskArgs args = *((const TaskArgs*)task->args);
  int rblk  = args.rblock;
  int Acols = args.Acols;
  int Bcols = args.Bcols;
  int rlo = p[0] * rblk;
  int rhi = (p[0] + 1) * rblk;
  //printf("(rblock=%d, Acols=%d, Bcols=%d)\n", rblk, Acols, Bcols);
  
  PtrMatrix AMat = get_raw_pointer(regions[0], rlo, rhi, 0, Acols);
  PtrMatrix BMat = get_raw_pointer(regions[1], rlo, rhi, 0, Bcols);

  PtrMatrix S(rblk, rblk);
  S.identity(); // initialize to identity matrix
  
  // assume V0'*u0 and V1'*u1 have the same number of rows
  assert(rblk%2==0);
  int r = rblk / 2;
  for (int i=0; i<r; i++) {
    for (int j=0; j<r; j++) {
      S(r+i, j) = AMat(i, j);
      S(i, r+j) = AMat(r+i, j);
    }
  }
  // set the right hand side
  for (int j=0; j<Bcols; j++) {
    for (int i=0; i<r; i++) {
      // switch BMat(i, j) with BMat(r+i, j)
      double temp = BMat(i, j);
      BMat(i, j) = BMat(r+i, j);
      BMat(r+i, j) = temp;
    }
  }  
  S.solve( BMat );
}
