#include "gemm_broadcast.hpp"

int GemmBroTask::TASKID;

GemmBroTask::GemmBroTask(Domain domain,
			 TaskArgument global_arg,
			 ArgumentMap arg_map,
			 Predicate pred,
			 bool must,
			 MapperID id,
			 MappingTagID tag)
  
  : IndexLauncher(TASKID, domain, global_arg,
		  arg_map, pred, must, id, tag) {}

void GemmBroTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <GemmBroTask::cpu_task>(AUTO_GENERATE_ID,
			    Processor::LOC_PROC, 
			    false,
			    true,
			    AUTO_GENERATE_ID,
			    TaskConfigOptions(true/*leaf*/),
			    "GemmBro_Solve");

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : GemmBro_Solve\n", TASKID);
#endif
}

void GemmBroTask::cpu_task(const Task *task,
			   const std::vector<PhysicalRegion> &regions,
			   Context ctx, HighLevelRuntime *runtime) {
}

