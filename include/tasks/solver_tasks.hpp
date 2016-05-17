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
#include "node_solve_region.hpp"
#include "gemm.hpp"
#include "gemm_inplace.hpp"
#include "gemm_reduce.hpp"
#include "gemm_broadcast.hpp"
#include "projector.hpp"
#include "reduce_add.hpp"

//#include "mapper.hpp"
//#include "spmd_mapper.hpp"
//#include "dist_mapper.hpp"

void register_solver_tasks();

void create_callback(Machine machine, HighLevelRuntime *rt,
		     const std::set<Processor> &local_procs);

void create_projector(Machine machine, HighLevelRuntime *rt,
		      const std::set<Processor> &local_procs);

#endif
