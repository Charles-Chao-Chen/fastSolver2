#include "clear_matrix.hpp"
#include "ptr_matrix.hpp"
#include "utility.hpp"

int ClearMatrixTask::TASKID;

ClearMatrixTask::ClearMatrixTask(Domain domain,
			       TaskArgument global_arg,
			       ArgumentMap arg_map,
			       Predicate pred,
			       bool must,
			       MapperID id,
			       MappingTagID tag)
  
  : IndexLauncher(TASKID, domain, global_arg,
		  arg_map, pred, must, id, tag) {}

void ClearMatrixTask::register_tasks(void)
{
  TASKID = Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(TASKID, "Clear_Matrix");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_leaf(true);
  Runtime::preregister_task_variant<ClearMatrixTask::cpu_task>(registrar, "cpu");

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : Clear_Matrix\n", TASKID);
#endif
}

void ClearMatrixTask::cpu_task(const Task *task,
			      const std::vector<PhysicalRegion> &regions,
			      Context ctx, Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->arglen == sizeof(TaskArgs));

  Point<1> p = task->index_point;
  //printf("point = %d\n", p[0]);

  const TaskArgs args = *((const TaskArgs*)task->args);
  int rblk  = args.rblock;
  int cols  = args.cols;
  double value = args.value;

  int rlo = p[0]*rblk;
  int rhi = (p[0] + 1) * rblk;
  //double *base = region_pointer(regions[0], rlo, rhi, 0, cols);
  PtrMatrix A = get_raw_pointer(regions[0], rlo, rhi, 0, cols);
  A.clear(value);
}
