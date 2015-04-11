#ifndef _init_matrix_hpp
#define _init_matrix_hpp

#include "legion.h"
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

class InitMatrixTask : public IndexLauncher {
public:
  struct TaskArgs {
    int rows;
    int cols;
  };
  InitMatrixTask(Domain domain,
		 TaskArgument global_arg,
		 ArgumentMap arg_map,
		 Predicate pred = Predicate::TRUE_PRED,
		 bool must = false,
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
