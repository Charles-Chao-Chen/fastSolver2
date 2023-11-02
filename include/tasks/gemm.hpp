#ifndef _gemm_hpp
#define _gemm_hpp

#include "legion.h"
using namespace Legion;

class GemmTask : public TaskLauncher {
public:
  struct TaskArgs {
    char transA;
    char transB;
    double alpha;
    double beta;
    int Arows;
    int Brows;
    int Crows;
    int AcolIdx;
    int BcolIdx;
    int Acols;
    int Bcols;
    int Ccols;
  };
  
  GemmTask(TaskArgument arg,
	   Predicate pred = Predicate::TRUE_PRED,
	   MapperID id = 0,
	   MappingTagID tag = 0);

  static void register_tasks(void);
  
  static int TASKID;
  
  static void
  cpu_task(const Task *task,
	   const std::vector<PhysicalRegion> &regions,
	   Context ctx, Runtime *runtime);
};

#endif
