#include "display_matrix.hpp"
#include "ptr_matrix.hpp"

#include "utility.hpp" // for FIELDID_V
#include <assert.h>

DisplayMatrixTask::TaskArgs::TaskArgs
(const std::string& name_, int rows_, int cols_) {
  strcpy(this->name, name_.c_str());
  this->rows = rows_;
  this->cols = cols_;
}

int DisplayMatrixTask::TASKID;

DisplayMatrixTask::DisplayMatrixTask
(TaskArgument arg, Predicate pred, MapperID id, MappingTagID tag)
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

void DisplayMatrixTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <DisplayMatrixTask::cpu_task>(AUTO_GENERATE_ID,
				  Processor::LOC_PROC, 
				  true,
				  false,
				  AUTO_GENERATE_ID,
				  TaskConfigOptions(true/*leaf*/),
				  "Display_Matrix");

  //#ifndef SHOW_REGISTER_TASKS
  printf("Register task %d : Display_Matrix\n", TASKID);
  //#endif
}

void DisplayMatrixTask::cpu_task
(const Task *task, const std::vector<PhysicalRegion> &regions,
 Context ctx, HighLevelRuntime *runtime) {
  
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->arglen == sizeof(TaskArgs));
  
  const TaskArgs args = *((const TaskArgs*)task->args);
  const char *name = args.name;
  const int   rows = args.rows;
  const int   cols = args.cols;
 
  //double *base = region_pointer(regions[0], 0, rows, 0, cols);    
  //PtrMatrix pMat(rows, cols, rows, base);
  PtrMatrix A = get_raw_pointer(regions[0], 0, rows, 0, cols);    
  A.display(name);
}



