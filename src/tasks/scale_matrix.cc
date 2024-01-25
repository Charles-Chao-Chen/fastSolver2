#include "scale_matrix.hpp"
#include "ptr_matrix.hpp"
#include "utility.hpp"

static Realm::Logger log_solver_tasks("solver_tasks");

int ScaleMatrixTask::TASKID;

ScaleMatrixTask::ScaleMatrixTask(Domain domain,
				 TaskArgument global_arg,
				 ArgumentMap arg_map,
				 MappingTagID tag,
				 Predicate pred,
				 bool must,
				 MapperID id)
  
  : IndexLauncher(TASKID, domain, global_arg,
		  arg_map, pred, must, id, tag) {}

void ScaleMatrixTask::register_tasks(void)
{
  TASKID = Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(TASKID, "Scale_Matrix");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_leaf(true);
  Runtime::preregister_task_variant<ScaleMatrixTask::cpu_task>(registrar, "cpu");

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : Scale_Matrix\n", TASKID);
#endif
}

void ScaleMatrixTask::cpu_task(const Task *task,
			      const std::vector<PhysicalRegion> &regions,
			      Context ctx, Runtime *runtime) {

  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->arglen == sizeof(TaskArgs));

  Point<1> p(task->index_point);
  //printf("point = %d\n", p[0]);

  log_solver_tasks.print("Inside scale matrix tasks.");

  const TaskArgs args = *((const TaskArgs*)task->args);
  int rblk  = args.rblock;
  int cols  = args.cols;
  double alpha = args.alpha;

  int rlo = (p[0]) * rblk;
  int rhi = (p[0] + 1) * rblk;
  PtrMatrix A = get_raw_pointer<LEGION_READ_WRITE>(regions[0], rlo, rhi, 0, cols);
  A.scale(alpha);
  //A.display("After scaling");
}
