#include "display_matrix.hpp"

#include "macros.hpp" // for FIELDID_V
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
 
  Rect<2> bounds, subrect;
  bounds.lo.x[0] = 0;
  bounds.hi.x[0] = rows - 1;
  bounds.lo.x[1] = 0;
  bounds.hi.x[1] = cols - 1;
  ByteOffset offsets[2];
  double *base = regions[0].get_field_accessor(FIELDID_V).template typeify<double>().template raw_rect_ptr<2>(bounds, subrect, offsets);
  assert(subrect == bounds);
#ifdef DEBUG_POINTERS
  printf("ptr = %p (%d, %d)\n", base, offsets[0].offset, offsets[1].offset);
#endif

  std::cout << name << ":"<< std::endl;
  for(int ri = 0; ri < rows; ri++) {
    for(int ci = 0; ci < cols; ci++) {
      double *value = base + ri * offsets[0] + ci * offsets[1];
      std::cout << *value << "\t";
    }
    std::cout << std::endl;
  }
}



