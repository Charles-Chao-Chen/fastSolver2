#include "leaf_solve.hpp"

int LeafSolveTask::TASKID;

LeafSolveTask::LeafSolveTask(Domain domain,
			     TaskArgument global_arg,
			     ArgumentMap arg_map,
			     Predicate pred,
			     bool must,
			     MapperID id,
			     MappingTagID tag)
  
  : IndexLauncher(TASKID, domain, global_arg,
		  arg_map, pred, must, id, tag) {}

void LeafSolveTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <LeafSolveTask::cpu_task>(AUTO_GENERATE_ID,
			    Processor::LOC_PROC, 
			    false,
			    true,
			    AUTO_GENERATE_ID,
			    TaskConfigOptions(true/*leaf*/),
			    "Leaf_Solve");

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : Leaf_Solve\n", TASKID);
#endif
}

void LeafSolveTask::cpu_task(const Task *task,
			     const std::vector<PhysicalRegion> &regions,
			     Context ctx, HighLevelRuntime *runtime) {
}



