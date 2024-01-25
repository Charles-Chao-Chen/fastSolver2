#ifndef _node_solve_region_hpp
#define _node_solve_region_hpp

#include "legion.h"
using namespace Legion;

class NodeSolveRegionTask : public TaskLauncher {
public:
  struct TaskArgs {
    int rank0;
    int rank1;
    int nRhs;
  };
  NodeSolveRegionTask(TaskArgument arg,
		      Predicate pred = Predicate::TRUE_PRED,
		      MapperID id = 0,
		      MappingTagID tag = 0);
  
  static int TASKID;

  static void register_tasks(void);

public:
  static void
  cpu_task(const Task *task,
	   const std::vector<PhysicalRegion> &regions,
	   Context ctx, Runtime *runtime);
};

#endif
