#ifndef _gemm_broadcast_hpp
#define _gemm_broadcast_hpp

#include "legion.h"
using namespace Legion;

class GemmBroTask : public IndexLauncher {
public:
  // the first member must be colorSize, which is referenced
  //  in the projector
  struct TaskArgs {
    int colorSize;
    int plevel;
    double alpha;
    char transa, transb;
    int Arblk, Brblk, Crblk;
    int Acols, Bcols, Ccols;
    int AcolIdx;
  };
  
  GemmBroTask(Domain domain,
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
