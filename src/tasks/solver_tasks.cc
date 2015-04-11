#include "solver_tasks.hpp"

void register_solver_tasks() {
  LeafSolveTask::register_tasks();
  NodeSolveTask::register_tasks();
  GemmRedTask::register_tasks();
  GemmBroTask::register_tasks();
  Add::register_operator();
  HighLevelRuntime::set_registration_callback(register_projector);

  InitMatrixTask::register_tasks();
  ZeroMatrixTask::register_tasks();
  DisplayMatrixTask::register_tasks();
}
