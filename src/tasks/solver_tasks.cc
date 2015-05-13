#include "solver_tasks.hpp"

void registration_callback(Machine machine, HighLevelRuntime *rt,
			   const std::set<Processor> &local_procs) {

  std::set<Processor>::const_iterator it = local_procs.begin();
  for (; it != local_procs.end(); it++) {
#if 0
    rt->replace_default_mapper
      (new SolverMapper(machine, rt, *it),*it);
#else
    rt->replace_default_mapper
      (new DistMapper(machine, rt, *it), *it);
#endif
  }
    
  rt->register_projection_functor
    (CONTRACTION, new Contraction(rt));
}

void register_solver_tasks() {
  InitMatrixTask::register_tasks();
  DenseBlockTask::register_tasks();
  AddMatrixTask::register_tasks();
  ClearMatrixTask::register_tasks();
  ScaleMatrixTask::register_tasks();
  DisplayMatrixTask::register_tasks();
  
  LeafSolveTask::register_tasks();
  NodeSolveTask::register_tasks();
  GemmRedTask::register_tasks();
  GemmBroTask::register_tasks();
  Add::register_operator();
  HighLevelRuntime::set_registration_callback(registration_callback);
}
