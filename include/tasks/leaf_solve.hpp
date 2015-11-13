#ifndef _leaf_solve_hpp
#define _leaf_solve_hpp

#include "legion.h"
using namespace LegionRuntime::HighLevel;

class LeafSolveTask : public IndexLauncher {
public:
  struct TaskArgs {
    int nrow;
    int nRhs;
    int rank;
    int nPart;
  };
  LeafSolveTask(Domain domain,
		TaskArgument global_arg,
		ArgumentMap arg_map,
		MappingTagID tag = 0,
		Predicate pred = Predicate::TRUE_PRED,
		bool must = false,
		MapperID id = 0);
  
  static int TASKID;

  static void register_tasks(void);

public:
  static void
  cpu_task(const Task *task,
	   const std::vector<PhysicalRegion> &regions,
	   Context ctx, HighLevelRuntime *runtime);
};

#endif
