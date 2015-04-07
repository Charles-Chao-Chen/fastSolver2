#include "node_solve.hpp"

int NodeSolveTask::TASKID;

NodeSolveTask::NodeSolveTask(Domain domain,
			     TaskArgument global_arg,
			     ArgumentMap arg_map,
			     Predicate pred,
			     bool must,
			     MapperID id,
			     MappingTagID tag)
  
  : IndexLauncher(TASKID, domain, global_arg,
		  arg_map, pred, must, id, tag) {}

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
}



