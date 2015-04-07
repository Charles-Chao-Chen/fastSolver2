#ifndef _solver_tasks_hpp
#define _solver_tasks_hpp

#include "projector.hpp"
#include "leaf_solve.hpp"
#include "node_solve.hpp"
#include "gemm_reduce.hpp"
#include "gemm_broadcast.hpp"
#include "reduce_add.hpp"

void register_solver_tasks() {
  LeafSolveTask::register_tasks();
  NodeSolveTask::register_tasks();
  GemmRedTask::register_tasks();
  GemmBroTask::register_tasks();
  Add::register_operator();
}

#endif
