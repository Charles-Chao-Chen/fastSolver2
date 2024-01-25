#ifndef _clear_matrix_hpp
#define _clear_matrix_hpp

#include "legion.h"
using namespace Legion;

class ClearMatrixTask : public IndexLauncher {
public:
  struct TaskArgs {
    int rblock;
    int cols;
    double value;
  };
  ClearMatrixTask(Domain domain,
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
	   Context ctx, Runtime *runtime);
};

#endif
