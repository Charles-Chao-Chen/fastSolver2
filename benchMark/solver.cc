#include <iostream>
#include <math.h>

// legion stuff
#include "legion.h"
using namespace LegionRuntime::HighLevel;

#include "matrix.hpp"  // for Matrix  class
#include "hmatrix.hpp" // for HMatrix class

enum {
  TOP_LEVEL_TASK_ID = 0,
  SPMD_TASK_ID = 1,
};

struct SPMDargs {
  std::vector<PhaseBarrier> reduction;
  std::vector<PhaseBarrier> node_solve;
};

void launch_solver_tasks
(int rank, int treelvl, int launchlvl, int niter, bool tracing,
 Context ctx, HighLevelRuntime *runtime) {

  // The number of processors should be 8 * #machines, i.e., 2^launchlvl
  // and the number of partitioning, i.e., the number of leaf nodes
  // should be 2^treelvl
  assert(treelvl >= launchlvl);

  // ======= Problem configuration =======
  // solve: A x = b where A = U * V' + D
  // =====================================
  int    base = 400, n = rank;
  bool   has_entry = false; //true;
  Matrix VMat(base, treelvl, n, has_entry);
  Matrix UMat(base, treelvl, n, has_entry);
  Matrix Rhs(base, treelvl, 1, has_entry);
  Vector DVec(base, treelvl, has_entry);

  // ================================================
  // generate random matrices, which is
  //  done in parallel
  // ================================================
  VMat.rand();
  UMat.rand();
  Rhs.rand();
  int mean = 1e3;
  DVec.rand(mean);

  // ========================================================
  // use the fast solver for a simple matrix U * V' + D,
  //  where the off-diagonal blocks are exactly low rank,
  //  so the solve should be accurate (up to round-off errors).
  // ========================================================

  // init tree
  int   nProc = pow(2,launchlvl);
  UTree uTree; uTree.init( nProc, UMat );
  VTree vTree; vTree.init( nProc, VMat );
  KTree kTree; kTree.init( nProc, UMat, VMat, DVec );

  // data partition
  uTree.partition( launchlvl, ctx, runtime );
  vTree.partition( launchlvl, ctx, runtime );
  kTree.partition( launchlvl, ctx, runtime );
  
  // init rhs
  uTree.init_rhs(Rhs, ctx, runtime, true/*wait*/);

  TraceID tID = 321;
  for (int it=0; it<niter; it++) {
    if (tracing) runtime->begin_trace(ctx, tID);
    
    // leaf solve: U = dense \ U
    kTree.solve( uTree.leaf(), vTree.leaf(), ctx, runtime );  

    for (int i=launchlvl; i>0; i--) {
      LMatrix& V = vTree.level(i);
      LMatrix& u = uTree.uMat_level(i);
      LMatrix& d = uTree.dMat_level(i);    
    
      // reduction operation
      int rows = pow(2, i)*V.cols();
      LMatrix VTu(rows, u.cols(), i-1, ctx, runtime);
      LMatrix VTd(rows, d.cols(), i-1, ctx, runtime);
      VTu.two_level_partition(ctx, runtime);
      VTd.two_level_partition(ctx, runtime);

      LMatrix::gemmRed('t', 'n', 1.0, V, u, 0.0, VTu, ctx, runtime );
      LMatrix::gemmRed('t', 'n', 1.0, V, d, 0.0, VTd, ctx, runtime );

      // form and solve the small linear system
      VTu.node_solve( VTd, ctx, runtime );
      
      // broadcast operation
      // d -= u * VTd
      if (it==0 && i==1) {
	LMatrix::gemmBro('n', 'n', -1.0, u, VTd, 1.0, d, ctx, runtime,
			 true/*wait*/ );
      } else {
	LMatrix::gemmBro('n', 'n', -1.0, u, VTd, 1.0, d, ctx, runtime );
      }
      std::cout<<"launched solver tasks at level: "<<i<<std::endl;
    }

    if (tracing) runtime->end_trace(ctx, tID);
  }
  
#ifdef SOLVER_RESIDULE
  // compute residule
  Matrix x = uTree.solution(ctx, runtime);
  Matrix err = Rhs - ( UMat * (VMat.T() * x) + DVec.multiply(x) );
  //err.display("err");
  std::cout << "Relative residual: " << err.norm() / Rhs.norm()
	    << std::endl;
#endif

  std::cout<<"Launching solver tasks complete."<<std::endl;
}

void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {
  
  // machine configuration
  int num_machines = 1;
  int num_cores_per_machine = 1;
  int task_level = (int)log2(num_cores_per_machine);

  // HODLR configuration
  int rank = 100;
  int leaf_size = 400;
  int matrix_level = task_level;

  // tracing configuration
  int niter = 1;
  bool tracing = false;

  // parse input arguments
  const InputArgs &command_args = HighLevelRuntime::get_input_args();
  if (command_args.argc > 1) {
    for (int i = 1; i < command_args.argc; i++) {
      if (!strcmp(command_args.argv[i],"-machine"))
	num_machines = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-core"))
	num_cores_per_machine = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-rank"))
	rank = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-leaf"))
	leaf_size = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-mtxlvl"))
	matrix_level = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-niter"))
	niter = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-tracing"))
	if (atoi(command_args.argv[++i]) != 0)
	  tracing = true;
    }
    assert(is_power_of_two(num_machines));
    assert(is_power_of_two(num_cores_per_machine));
    assert(rank                  > 0);
    assert(leaf_size             > 0);
    assert(matrix_level          >= task_level);
    assert(niter                 > 0);
  }
  std::cout<<"\n========================"
           <<"\nRunning fast solver..."
           <<"\n---------------------"
	   <<"\n# machines: "<<num_machines
	   <<"\n# cores/machine: "<<num_cores_per_machine
           <<"\n---------------------"
	   <<"\noff-diagonal rank: "<<rank
	   <<"\nleaf size: "<<leaf_size
	   <<"\nmatrix level: "<<matrix_level
           <<"\n---------------------"
	   <<"\niteration number: "<<niter
	   <<"\nlegion tracing: "<<std::boolalpha<<tracing
           <<"\n========================\n"
	   <<std::endl;

  // create phase barriers
  std::vector<SPMDargs> args(num_machines);

  int spmd_tree_level = (int)log2(num_machines);
  for (int l=0; l<spmd_tree_level; l++) {
    int num_barriers = (int)pow(2, l); 
    int num_shards_per_barrier = (int)pow(2, spmd_tree_level-l);
    PhaseBarrier pb_reduction = runtime->create_phase_barrier(ctx, num_shards_per_barrier);
    PhaseBarrier pb_node_solve = runtime->create_phase_barrier(ctx, 1);
    std::vector<PhaseBarrier> barrier_reduction(num_barriers, pb_reduction);
    std::vector<PhaseBarrier> barrier_node_solve(num_barriers, pb_node_solve);
    for (int shard=0; shard<num_machines; shard++) {
      int barrier_idx = shard / num_shards_per_barrier;
      args[shard].reduction.push_back(barrier_reduction[barrier_idx]);
      args[shard].node_solve.push_back(barrier_node_solve[barrier_idx]);
    }
  }

  // create ghost regions
  Point<2> lo = make_point(0, 0);
  Point<2> hi = make_point(2*rank-1, 2*rank-1);
  Rect<2>  rect(lo, hi);
  IndexSpace is = runtime->create_index_space(ctx,
                          Domain::from_rect<2>(rect));
  runtime->attach_name(is, "ghost_is");
  FieldSpace fs = runtime->create_field_space(ctx);
  runtime->attach_name(fs, "ghost_fs");
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double), FID_GHOST);
    runtime->attach_name(fs, FID_GHOST, "GHOST");
  }

  std::vector<TaskLauncher> spmd_tasks;
  for (int i=0; i<num_machines; i++) {
    spmd_tasks.push_back
      (TaskLauncher(SPMD_TASK_ID, TaskArgument(&args[i], sizeof(SPMDargs))));
  }

  for (int l=0; l<spmd_tree_level; l++) {
    int num_ghosts = (int)pow(2, l);
    int num_shards_per_ghost = (int)pow(2, spmd_tree_level-l);
    std::vector<LogicalRegion> ghosts;
    for (int i=0; i<num_ghosts; i++) {
      ghosts.push_back(runtime->create_logical_region(ctx, is, fs));
    }
    for (int shard=0; shard<num_machines; shard++) {
      int idx = shard / num_shards_per_ghost;
      spmd_tasks[shard].add_region_requirement
	(RegionRequirement(ghosts[idx],READ_WRITE,SIMULTANEOUS,ghosts[idx]));
      spmd_tasks[shard].region_requirements[l].flags |= NO_ACCESS_FLAG;
      spmd_tasks[shard].add_index_requirement
	(IndexSpaceRequirement(is, NO_MEMORY, is));
      spmd_tasks[shard].add_field(l, FID_GHOST);
    }
  }
  
  // create SPMD tasks
  MustEpochLauncher must_epoch_launcher;
  for (int shard=0; shard<num_machines; shard++) {
    DomainPoint point(shard);
    must_epoch_launcher.add_single_task(point, spmd_tasks[shard]);
  }  

  runtime->execute_must_epoch(ctx, must_epoch_launcher);
  runtime->destroy_index_space(ctx, is);
  runtime->destroy_field_space(ctx, fs);
}

int main(int argc, char *argv[]) {
  // register top level task
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(
    TOP_LEVEL_TASK_ID,   /* task id */
    Processor::LOC_PROC, /* cpu */
    true,  /* single */
    false, /* index  */
    AUTO_GENERATE_ID,
    TaskConfigOptions(false /*leaf task*/),
    "master-task"
  );

  // register solver tasks
  register_solver_tasks();

  // register mapper
  HighLevelRuntime::set_registration_callback(registration_callback);

  // start legion master task
  return HighLevelRuntime::start(argc, argv);
}

