#ifndef _solver_tasks_hpp
#define _solver_tasks_hpp

#include "leaf_solve.hpp"
#include "node_solve.hpp"
#include "gemm_reduce.hpp"
#include "gemm_broadcast.hpp"
#include "projector.hpp"
#include "reduce_add.hpp"

#include "init_matrix.hpp"
#include "zero_matrix.hpp"
#include "display_matrix.hpp"

void register_solver_tasks();

#endif
