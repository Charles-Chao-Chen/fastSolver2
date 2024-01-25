#ifndef _scale_matrix_hpp
#define _scale_matrix_hpp

#include "legion.h"
using namespace Legion;

class ScaleMatrixTask : public IndexLauncher {
public:
  struct TaskArgs {
    int rblock;
    int cols;
    double alpha;
  };
  ScaleMatrixTask(Domain domain,
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
	   Context ctx, Runtime *runtime);
};

#endif
