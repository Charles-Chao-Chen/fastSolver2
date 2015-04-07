#ifndef _gemm_broadcast_hpp
#define _gemm_broadcast_hpp

#include "legion.h"
using namespace LegionRuntime::HighLevel;

class GemmBroTask : public IndexLauncher {
public:
  
  struct TaskArgs {
    int    colorSize; // used in projector
    double alpha;
    double beta;
  };
  
  GemmBroTask(Domain domain,
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
