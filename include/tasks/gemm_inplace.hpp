#ifndef _gemm_inplace_hpp
#define _gemm_inplace_hpp

#include "legion.h"
using namespace LegionRuntime::HighLevel;

class GemmInplaceTask : public TaskLauncher {
public:
  struct TaskArgs {
    char transa;
    char transb;
    double alpha;
    double beta;
  };
  
  GemmInplaceTask
  (TaskArgument arg, Predicate pred = Predicate::TRUE_PRED,
   MapperID id = 0, MappingTagID tag = 0);

  static void register_tasks(void);
  
  static int TASKID;
  
  static void
  cpu_task(const Task *task,
	   const std::vector<PhysicalRegion> &regions,
	   Context ctx, HighLevelRuntime *runtime);
};

#endif
