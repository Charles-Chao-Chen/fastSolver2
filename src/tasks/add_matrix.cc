#include "add_matrix.hpp"
#include "ptr_matrix.hpp"
#include "utility.hpp"

int AddMatrixTask::TASKID;

AddMatrixTask::AddMatrixTask(Domain domain,
			     TaskArgument global_arg,
			     ArgumentMap arg_map,
			     Predicate pred,
			     bool must,
			     MapperID id,
			     MappingTagID tag)
  
  : IndexLauncher(TASKID, domain, global_arg,
		  arg_map, pred, must, id, tag) {}

void AddMatrixTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <AddMatrixTask::cpu_task>(AUTO_GENERATE_ID,
			    Processor::LOC_PROC, 
			    false,
			    true,
			    AUTO_GENERATE_ID,
			    TaskConfigOptions(true/*leaf*/),
			    "add_matrix");

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : add_matrix\n", TASKID);
#endif
}

void AddMatrixTask::cpu_task(const Task *task,
			     const std::vector<PhysicalRegion> &regions,
			     Context ctx, HighLevelRuntime *runtime) {

  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  assert(task->arglen == sizeof(TaskArgs));

  Point<1> p = task->index_point.get_point<1>();
  printf("point = %d\n", p[0]);

  const TaskArgs args = *((const TaskArgs*)task->args);
  int rblk = args.nrow;
  int cols = args.cols;
  double alpha = args.alpha;
  double beta  = args.beta;

  int rlo = p[0]*rblk;
  int rhi = (p[0] + 1) * rblk;
  
  PtrMatrix AMat = get_raw_pointer(regions[0], rlo, rhi, 0, cols);
  PtrMatrix BMat = get_raw_pointer(regions[1], rlo, rhi, 0, cols);
  PtrMatrix CMat = get_raw_pointer(regions[2], rlo, rhi, 0, cols);
  PtrMatrix::add(alpha, AMat, beta, BMat, CMat);
}
