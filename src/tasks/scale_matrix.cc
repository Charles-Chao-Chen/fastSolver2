#include "scale_matrix.hpp"
#include "ptr_matrix.hpp"
#include "utility.hpp"

int ScaleMatrixTask::TASKID;

ScaleMatrixTask::ScaleMatrixTask(Domain domain,
			       TaskArgument global_arg,
			       ArgumentMap arg_map,
			       Predicate pred,
			       bool must,
			       MapperID id,
			       MappingTagID tag)
  
  : IndexLauncher(TASKID, domain, global_arg,
		  arg_map, pred, must, id, tag) {}

void ScaleMatrixTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <ScaleMatrixTask::cpu_task>(AUTO_GENERATE_ID,
			       Processor::LOC_PROC, 
			       false,
			       true,
			       AUTO_GENERATE_ID,
			       TaskConfigOptions(true/*leaf*/),
			       "Scale_Matrix");

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : Scale_Matrix\n", TASKID);
#endif
}

void ScaleMatrixTask::cpu_task(const Task *task,
			      const std::vector<PhysicalRegion> &regions,
			      Context ctx, HighLevelRuntime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->arglen == sizeof(TaskArgs));

  Point<1> p = task->index_point.get_point<1>();
  //printf("point = %d\n", p[0]);

  const TaskArgs args = *((const TaskArgs*)task->args);
  int rblk  = args.rblock;
  int cols  = args.cols;
  double alpha = args.alpha;

  int rlo = (p[0]) * rblk;
  int rhi = (p[0] + 1) * rblk;
  PtrMatrix A = get_raw_pointer(regions[0], rlo, rhi, 0, cols);
  A.scale(alpha);
  //A.display("After scaling");
}
