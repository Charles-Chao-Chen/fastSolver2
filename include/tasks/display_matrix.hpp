#ifndef _display_matrix_hpp
#define _display_matrix_hpp

#include "legion.h"
using namespace Legion;

#include <string>

class DisplayMatrixTask : public TaskLauncher {
public:
  struct TaskArgs {
    TaskArgs(const std::string&, int, int, int);
    char name[20];
    int  rows;
    int  cols;
    int  colIdx;
  };
  DisplayMatrixTask(TaskArgument global_arg,
		    Predicate pred = Predicate::TRUE_PRED,
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
