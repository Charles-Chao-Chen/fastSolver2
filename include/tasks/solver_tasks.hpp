#ifndef _solver_tasks_hpp
#define _solver_tasks_hpp

#include "init_matrix.hpp"
#include "dense_block.hpp"
#include "add_matrix.hpp"
#include "clear_matrix.hpp"
#include "scale_matrix.hpp"
#include "display_matrix.hpp"

#include "leaf_solve.hpp"
#include "node_solve.hpp"
#include "gemm_reduce.hpp"
#include "gemm_broadcast.hpp"
#include "projector.hpp"
#include "reduce_add.hpp"

#include "mapper.hpp"

void register_solver_tasks();

void registration_callback(Machine machine, HighLevelRuntime *rt,
			   const std::set<Processor> &local_procs);

#endif
