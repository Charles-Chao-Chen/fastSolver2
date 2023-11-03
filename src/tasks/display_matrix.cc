#include "display_matrix.hpp"
#include "ptr_matrix.hpp"

#include "utility.hpp" // for FIELDID_V
#include <assert.h>

DisplayMatrixTask::TaskArgs::TaskArgs
(const std::string& name_, int rows_, int cols_, int begin) {
  strcpy(this->name, name_.c_str());
  this->rows = rows_;
  this->cols = cols_;
  this->colIdx = begin;
}

int DisplayMatrixTask::TASKID;

DisplayMatrixTask::DisplayMatrixTask
(TaskArgument arg, Predicate pred, MapperID id, MappingTagID tag)
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

void DisplayMatrixTask::register_tasks(void)
{
  TASKID = Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(TASKID, "Display_Matrix");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_leaf(true);
  Runtime::preregister_task_variant<DisplayMatrixTask::cpu_task>(registrar, "cpu");

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : Display_Matrix\n", TASKID);
#endif
}

void DisplayMatrixTask::cpu_task
(const Task *task, const std::vector<PhysicalRegion> &regions,
 Context ctx, Runtime *runtime) {
  
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->arglen == sizeof(TaskArgs));
  
  const TaskArgs args = *((const TaskArgs*)task->args);
  const char *name = args.name;
  const int   rows = args.rows;
  const int   cols = args.cols;
  const int   colIdx = args.colIdx;
  
  PtrMatrix A = get_raw_pointer(regions[0], 0, rows, colIdx, colIdx+cols);
  A.display(name);
}



