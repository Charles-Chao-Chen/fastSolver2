#ifndef _dense_block_hpp
#define _dense_block_hpp

#include "legion.h"
using namespace Legion;

struct ThreeSeeds {
  long uSeed;
  long vSeed;
  long dSeed;
};

class DenseBlockTask : public IndexLauncher {
public:
  struct TaskArgs {
    int size;
    int rank;
    int offset;
  };
  DenseBlockTask(Domain domain,
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
