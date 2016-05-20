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

const int MAX_TREE_LEVEL = 10;

struct SPMDargs {
  PhaseBarrier reduction[MAX_TREE_LEVEL];
  PhaseBarrier node_solve[MAX_TREE_LEVEL];
  int leaf_size;
  int rank;
  int nRhs;
  int spmd_level;
  int my_matrix_level;
  int my_task_level;
};

bool is_master_task(int point, int current_level, int total_level) {
  assert(0<= current_level && current_level < total_level);
  return (point % (int) pow(2, total_level - current_level) == 0);
}

LMatrix create_local_region(LogicalRegion ghost, int rows, int cols,
			    Context ctx, HighLevelRuntime *runtime) {
  IndexSpace is = ghost.get_index_space();
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double), FIELDID_V);
  }
  LogicalRegion region = 
    runtime->create_logical_region(ctx, is, fs);  
  return LMatrix(rows, cols, region, is, fs);
}

LMatrix create_legion_matrix(LogicalRegion region, int rows, int cols) {
  return LMatrix(region, rows, cols);
}

void spmd_fast_solver(const Task *task,
		      const std::vector<PhysicalRegion> &regions,
		      Context ctx, HighLevelRuntime *runtime) {

  int spmd_point = task->index_point.get_index();
  std::cout<<"Inside spmd_task["<<spmd_point<<"]\n";

  runtime->unmap_all_regions(ctx);
  SPMDargs *args       = (SPMDargs*)task->args;
  int leaf_size        = args->leaf_size;
  int rank             = args->rank;
  int nRhs             = args->nRhs;
  int spmd_level       = args->spmd_level;
  int matrix_level     = args->my_matrix_level;
  int task_level       = args->my_task_level;
  
  // ======= Problem configuration =======
  // solve: A x = b where A = U * V' + D
  // =====================================
  bool   has_entry = false; //true;
  Matrix VMat(leaf_size, matrix_level, rank, has_entry);
  Matrix UMat(leaf_size, matrix_level, rank, has_entry);
  Matrix Rhs (leaf_size, matrix_level, nRhs, has_entry);
  Vector DVec(leaf_size, matrix_level,       has_entry);

  // randomly generate entries
  VMat.rand();
  UMat.rand();
  Rhs.rand();
  int mean = 1e3;
  DVec.rand(mean);

  // init tree
  int global_tree_level = spmd_level+matrix_level;
  UTree uTree; uTree.init( global_tree_level, UMat, ctx, runtime );
  VTree vTree; vTree.init( global_tree_level, VMat, ctx, runtime );
  KTree kTree; kTree.init( matrix_level, UMat, VMat, DVec, ctx, runtime );
  
  // data partition
  uTree.horizontal_partition( task_level, ctx, runtime );
  vTree.horizontal_partition( task_level, ctx, runtime );
  kTree.horizontal_partition( task_level, ctx, runtime );
  
  // init rhs and wait
  uTree.init_rhs(Rhs, ctx, runtime, true/*wait*/);

  // computation starts now
  
  // leaf solve: U = dense \ U
  kTree.solve( uTree.leaf(), vTree.leaf(), ctx, runtime );  

  // solve on every machine
  for (int i=task_level-1; i>=0; i--) {
    int tree_level = i + spmd_level;
    LMatrix& V = vTree.level_new(tree_level);
    LMatrix& u = uTree.uMat_level_new(tree_level);
    LMatrix& d = uTree.dMat_level_new(tree_level);
    
    // reduction operation
    int rows = pow(2, i+1)*V.cols();    
    LMatrix VTu(rows, u.cols(), i, ctx, runtime);
    LMatrix VTd(rows, d.cols(), i, ctx, runtime);
    VTu.two_level_partition(ctx, runtime);
    VTd.two_level_partition(ctx, runtime);

    LMatrix::gemmRed('t', 'n', 1.0, V, u, 0.0, VTu, ctx, runtime );
    LMatrix::gemmRed('t', 'n', 1.0, V, d, 0.0, VTd, ctx, runtime );

    // form and solve the small linear system
    VTu.node_solve( VTd, ctx, runtime );
      
    // broadcast operation
    // d -= u * VTd
    LMatrix::gemmBro('n', 'n', -1.0, u, VTd, 1.0, d, ctx, runtime );
    std::cout<<"launched solver tasks at level: "<<tree_level<<std::endl;
  }

  // spmd level
  int ghost_idx = 0;
  for (int l=spmd_level-1; l>=0; l--) {
    LMatrix& V = vTree.level_new(l);
    LMatrix& u = uTree.uMat_level_new(l);
    LMatrix& d = uTree.dMat_level_new(l);    

    // compute local results
    LogicalRegion VTu_ghost = task->regions[ghost_idx].region;
    LogicalRegion VTd_ghost = task->regions[ghost_idx+1].region;
    LMatrix VTu = create_local_region(VTu_ghost, rank, rank, ctx, runtime);
    LMatrix VTd = create_local_region(VTd_ghost, rank, nRhs+l*rank, ctx, runtime);

    LMatrix::gemm('t', 'n', 1.0, V, u, 0.0, VTu, ctx, runtime );
    LMatrix::gemm('t', 'n', 1.0, V, d, 0.0, VTd, ctx, runtime );
    
    // copy
    CopyLauncher  cp_VTu;
    CopyLauncher  cp_VTd;
    LogicalRegion VTu_local = VTu.logical_region();
    LogicalRegion VTd_local = VTd.logical_region();
    cp_VTu.add_copy_requirements
      (RegionRequirement(VTu_local, READ_ONLY, EXCLUSIVE, VTu_local),
       RegionRequirement(VTu_ghost, REDOP_ADD, EXCLUSIVE, VTu_ghost));
    cp_VTd.add_copy_requirements
      (RegionRequirement(VTd_local, READ_ONLY, EXCLUSIVE, VTd_local),
       RegionRequirement(VTd_ghost, REDOP_ADD, EXCLUSIVE, VTd_ghost));
    cp_VTu.add_src_field(0, FIELDID_V);
    cp_VTu.add_dst_field(0, FIELDID_V);
    cp_VTd.add_src_field(0, FIELDID_V);
    cp_VTd.add_dst_field(0, FIELDID_V);
    assert(args->reduction[l]!=PhaseBarrier());
    cp_VTu.add_arrival_barrier(args->reduction[l]);
    cp_VTd.add_arrival_barrier(args->reduction[l]);
    runtime->issue_copy_operation(ctx, cp_VTu);
    runtime->issue_copy_operation(ctx, cp_VTd);
    
    // node solve
    if (is_master_task(spmd_point, l, spmd_level)) {
      LMatrix VTu0 = create_legion_matrix(VTu_ghost,rank,rank);
      LMatrix VTd0 = create_legion_matrix(VTd_ghost,rank,nRhs+rank*l);
      LogicalRegion VTu1_ghost = task->regions[ghost_idx+2].region;
      LogicalRegion VTd1_ghost = task->regions[ghost_idx+3].region;
      LMatrix VTu1 = create_legion_matrix(VTu1_ghost,rank,rank);
      LMatrix VTd1 = create_legion_matrix(VTd1_ghost,rank,nRhs+rank*l);
      args->reduction[l] = 
	runtime->advance_phase_barrier(ctx, args->reduction[l]);
      assert(args->node_solve[l]!=PhaseBarrier());
      LMatrix::node_solve( VTu0, VTu1, VTd0, VTd1,
			   args->reduction[l], args->node_solve[l],
			   ctx, runtime );
      ghost_idx += 4;
    }

    // copy data
    else {
      args->node_solve[l] = 
	runtime->advance_phase_barrier(ctx, args->node_solve[l]);
      CopyLauncher cp_node_solve;
      cp_node_solve.add_copy_requirements
	(RegionRequirement(VTd_ghost, READ_ONLY, EXCLUSIVE, VTd_ghost),
	 RegionRequirement(VTd_local, WRITE_DISCARD, EXCLUSIVE, VTd_local));
      cp_node_solve.add_src_field(0, FIELDID_V);
      cp_node_solve.add_dst_field(0, FIELDID_V);
      cp_node_solve.add_wait_barrier(args->node_solve[l]);
      runtime->issue_copy_operation(ctx, cp_node_solve);
      ghost_idx += 2;
    }

    // local update: d -= u * VTd
    LMatrix VTd_lmtx;
    if (is_master_task(spmd_point, l, spmd_level)) {
      VTd_lmtx = create_legion_matrix(VTd_ghost,rank,nRhs+rank*l);    
    }
    else {
      VTd_lmtx = create_legion_matrix(VTd_local,rank,nRhs+rank*l);    
    }
    LMatrix::gemm_inplace('n', 'n', -1.0, u, VTd_lmtx, 1.0, d, ctx, runtime );
    std::cout<<"launched solver tasks at level: "<<l<<std::endl;
  }
  // check residule  
}

void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {
  
  // machine configuration
  int num_machines = 1;
  int num_cores_per_machine = 1;
 
  // HODLR configuration
  int rank = 100;
  int leaf_size = 400;
  int matrix_level = 1;

  // right hand side
  const int nRhs = 1;

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
    }
    assert(is_power_of_two(num_machines));
    assert(is_power_of_two(num_cores_per_machine));
    assert(rank         > 0);
    assert(leaf_size    > 0);
  }
  int spmd_level = (int)log2(num_machines);
  int task_level = (int)log2(num_cores_per_machine);
  assert(spmd_level<=MAX_TREE_LEVEL);
  if(matrix_level < task_level+spmd_level) {
    matrix_level  = task_level+spmd_level;
    std::cout<<"--------------------------------------------------"<<std::endl
	     <<"Warning: matrix level is raised up to its minimum!"<<std::endl
	     <<"--------------------------------------------------"<<std::endl;
  }  
  std::cout<<"\n========================"
           <<"\nRunning fast solver..."
           <<"\n---------------------"
	   <<"\n# machines: "<<num_machines
	   <<", level: "<<spmd_level
	   <<"\n# cores/machine: "<<num_cores_per_machine
    	   <<", level: "<<task_level
           <<"\n---------------------"
	   <<"\noff-diagonal rank: "<<rank
	   <<"\nleaf size: "<<leaf_size
	   <<"\nmatrix level: "<<matrix_level
           <<"\n========================\n"
	   <<std::endl;

  // create phase barriers
  SPMDargs arg;
  arg.leaf_size = leaf_size;
  arg.rank = rank;
  arg.nRhs = nRhs;
  arg.spmd_level = spmd_level;
  arg.my_matrix_level = matrix_level - spmd_level;
  arg.my_task_level = task_level;
  std::vector<SPMDargs> args(num_machines, arg);
  for (int l=0; l<spmd_level; l++) {
    int num_barriers = (int)pow(2, l); 
    int num_shards_per_barrier = (int)pow(2, spmd_level-l);
    std::vector<PhaseBarrier> barrier_reduction;
    std::vector<PhaseBarrier> barrier_node_solve;
    for (int i=0; i<num_barriers; i++) {
      barrier_reduction.push_back
        (runtime->create_phase_barrier(ctx,2*num_shards_per_barrier/*VTu,VTd*/));
      barrier_node_solve.push_back
        (runtime->create_phase_barrier(ctx,1));
    }
    for (int shard=0; shard<num_machines; shard++) {
      int barrier_idx = shard / num_shards_per_barrier;
      args[shard].reduction[l] = barrier_reduction[barrier_idx];
      args[shard].node_solve[l] = barrier_node_solve[barrier_idx];
    }
  }
  
  // create spmd tasks  
  std::vector<TaskLauncher> spmd_tasks;
  for (int i=0; i<num_machines; i++) {
    spmd_tasks.push_back
      (TaskLauncher(SPMD_TASK_ID, TaskArgument(&args[i], sizeof(SPMDargs))));
  }
  
  // create ghost regions: VTu0, VTu1(r x r) and VTd0, VTd1(r x .)
  Point<2> lo = make_point(0, 0);
  Point<2> hi = make_point(rank-1, rank-1);
  Rect<2>  rect(lo, hi);
  IndexSpace VTu_is = runtime->create_index_space
    (ctx, Domain::from_rect<2>(rect));
  runtime->attach_name(VTu_is, "VTu_ghost_is");
  FieldSpace fs = runtime->create_field_space(ctx);
  runtime->attach_name(fs, "VTu_ghost_fs");
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double), FIELDID_V);
    runtime->attach_name(fs, FIELDID_V, "GHOST");
  }
  // go upward, this order should be consistant with task
  // launches, so the right ghost region can be referred to.
  for (int l=spmd_level-1; l>=0; l--) {
    int num_ghosts = (int)pow(2, l);
    int num_shards_per_ghost = (int)pow(2, spmd_level-l);
    std::vector<std::vector<LogicalRegion> > VTu_ghosts(num_ghosts);
    std::vector<std::vector<LogicalRegion> > VTd_ghosts(num_ghosts);
    for (int i=0; i<num_ghosts; i++) {
      // VTu0
      VTu_ghosts[i].push_back(runtime->create_logical_region(ctx, VTu_is, fs));
      // VTu1
      VTu_ghosts[i].push_back(runtime->create_logical_region(ctx, VTu_is, fs));
    }
    // create VTd regions
    Point<2> lo = make_point(0, 0);
    Point<2> hi = make_point(rank-1, nRhs+l*rank-1);
    Rect<2>  rect(lo, hi);
    IndexSpace VTd_is = runtime->create_index_space
      (ctx, Domain::from_rect<2>(rect));
    for (int i=0; i<num_ghosts; i++) {
      // VTd0
      VTd_ghosts[i].push_back(runtime->create_logical_region(ctx, VTd_is, fs));
      // VTd1
      VTd_ghosts[i].push_back(runtime->create_logical_region(ctx, VTd_is, fs));
    }
    for (int shard=0; shard<num_machines; shard++) {
      int idx    = shard / num_shards_per_ghost;
      int subidx = shard % num_shards_per_ghost / (num_shards_per_ghost/2);
      assert(subidx==0||subidx==1);

      // master task owns all ghost regions
      if (is_master_task(shard, l, spmd_level)) {
	assert(subidx==0);
	for (int i=0; i<2; i++) {
	  LogicalRegion VTu = VTu_ghosts[idx][i];
	  LogicalRegion VTd = VTd_ghosts[idx][i];
	  RegionRequirement VTu_req(VTu,READ_WRITE,SIMULTANEOUS,VTu);
	  RegionRequirement VTd_req(VTd,READ_WRITE,SIMULTANEOUS,VTd);
	  VTu_req.add_field(FIELDID_V);
	  VTd_req.add_field(FIELDID_V);
	  spmd_tasks[shard].add_region_requirement(VTu_req);
	  spmd_tasks[shard].add_region_requirement(VTd_req);
	}
      }

      // declare no access to ghost regions
      else {
	LogicalRegion VTu = VTu_ghosts[idx][subidx];
	LogicalRegion VTd = VTd_ghosts[idx][subidx];
	RegionRequirement VTu_req(VTu,READ_WRITE,SIMULTANEOUS,VTu);
	RegionRequirement VTd_req(VTd,READ_WRITE,SIMULTANEOUS,VTd);
	VTu_req.flags = NO_ACCESS_FLAG;
	VTd_req.flags = NO_ACCESS_FLAG;
	VTu_req.add_field(FIELDID_V);
	VTd_req.add_field(FIELDID_V);
	spmd_tasks[shard].add_region_requirement(VTu_req);
	spmd_tasks[shard].add_region_requirement(VTd_req);
      }
    }
  }
  
  // create must_epoch_launcher
  MustEpochLauncher must_epoch_launcher;
  for (int shard=0; shard<num_machines; shard++) {
    DomainPoint point(shard);
    must_epoch_launcher.add_single_task(point, spmd_tasks[shard]);
  }  
  FutureMap fm = runtime->execute_must_epoch(ctx, must_epoch_launcher);
  fm.wait_all_results();
  runtime->destroy_index_space(ctx, VTu_is);
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
  HighLevelRuntime::register_legion_task<spmd_fast_solver>(SPMD_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(), "spmd");

  // register solver tasks
  register_solver_tasks();

  // start legion master task
  return HighLevelRuntime::start(argc, argv);
}


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
  UTree uTree; uTree.init( UMat );
  VTree vTree; vTree.init( VMat );
  KTree kTree; kTree.init( UMat, VMat, DVec );

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

