#ifndef _gemm_reduce_hpp
#define _gemm_reduce_hpp

#include "legion.h"
using namespace LegionRuntime::HighLevel;

class GemmRedTask : public IndexLauncher {
public:
  
  struct TaskArgs {
    int    colorSize; // used in projector
    double alpha;
    double beta;
  };
  
  GemmRedTask(Domain domain,
	      TaskArgument global_arg,
	      ArgumentMap arg_map,
	      Predicate pred = Predicate::TRUE_PRED,
	      bool must = false,
	      MapperID id = 0,
	      MappingTagID tag = 0);

  static void register_tasks(void);
  
  static int TASKID;
  
  static void
  cpu_task(const Task *task,
	   const std::vector<PhysicalRegion> &regions,
	   Context ctx, HighLevelRuntime *runtime);
};

#endif
