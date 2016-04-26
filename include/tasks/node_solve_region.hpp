#ifndef _node_solve_region_hpp
#define _node_solve_region_hpp

#include "legion.h"
using namespace LegionRuntime::HighLevel;

class NodeSolveRegionTask : public TaskLauncher {
public:
  struct TaskArgs {
    int rblock;
    int Acols;
    int Bcols;
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
	   Context ctx, HighLevelRuntime *runtime);
};

#endif
