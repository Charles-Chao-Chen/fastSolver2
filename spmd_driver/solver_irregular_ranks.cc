#include <iostream>
#include <math.h>
#include <unistd.h>

// legion stuff
#include "legion.h"
using namespace Legion;

#include "matrix.hpp"  // for Matrix  class
#include "hmatrix.hpp" // for HMatrix class

enum {
  TOP_LEVEL_TASK_ID = 0,
  SPMD_TASK_ID = 1,
};

const int MAX_TREE_LEVEL = 20;

struct SPMDargs {
  PhaseBarrier reduction[MAX_TREE_LEVEL];
  PhaseBarrier node_solve[MAX_TREE_LEVEL];
  int leaf_size;
  int rank0;
  int rank1;
  int nRhs;
  int spmd_level;
  int my_matrix_level;
  int my_task_level;
};

bool is_master_task(int point, int current_level, int total_level) {
  assert(0<= current_level && current_level < total_level);
  return (point % (int) pow(2, total_level - current_level) == 0);
}

/*
LMatrix create_local_region(LogicalRegion ghost, int rows, int cols,
			    Context ctx, Runtime *runtime) {
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
*/

LogicalRegion create_local_region(LogicalRegion ghost,
			    Context ctx, Runtime *runtime) {
  IndexSpace is = ghost.get_index_space();
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double), FIELDID_V);
  }
  return runtime->create_logical_region(ctx, is, fs);  
}

LMatrix create_legion_matrix(LogicalRegion region, int rows, int cols) {
  return LMatrix(region, rows, cols);
}

LMatrix create_legion_matrix(LogicalRegion region,
			    Context ctx, Runtime *runtime) {
  Domain dom = runtime->get_index_space_domain(ctx,region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();
  int rows = rect.dim_size(0);
  int cols = rect.dim_size(1);
  return LMatrix(region, rows, cols);
}

void spmd_fast_solver(const Task *task,
		      const std::vector<PhysicalRegion> &regions,
		      Context ctx, Runtime *runtime) {

  char hostname[100];
  gethostname(hostname, 100);
  int spmd_point = task->index_point.get_index();
  std::cout<<"spmd_task["<<spmd_point<<"] is running on machine "
	   <<hostname<<std::endl;

  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  assert(task->arglen == sizeof(SPMDargs));
  
  runtime->unmap_all_regions(ctx);
  SPMDargs *args       = (SPMDargs*)task->args;
  int leaf_size        = args->leaf_size;
  int nRhs             = args->nRhs;
  int spmd_level       = args->spmd_level;
  int matrix_level     = args->my_matrix_level;
  int task_level       = args->my_task_level;
  
  int rank = 0, rankp = 0; // rank prime corresponds to the sibling
  if (spmd_point == 0) {
    rank  = args->rank0;
    rankp = args->rank1;
  }
  if (spmd_point == 1) {
    rank  = args->rank1;
    rankp = args->rank0;
  }
  
  // ======= Problem configuration =======
  // solve: A x = b where A = U * V' + D
  // =====================================
  bool   has_entry = false; //true;
  Matrix VMat(leaf_size, matrix_level, rank, has_entry);
  Matrix UMat(leaf_size, matrix_level, rank, has_entry);
  Matrix Rhs (leaf_size, matrix_level, nRhs, has_entry);
  Vector DVec(leaf_size, matrix_level,       has_entry);
  Matrix VMatp(leaf_size, matrix_level, rankp, has_entry);

  // randomly generate entries
  VMat.rand();
  VMatp.rand();
  UMat.rand();
  Rhs.rand();
  int mean = 1e3;
  DVec.rand(mean);

  // init tree
  int global_tree_level = spmd_level+matrix_level;
  UTree uTree; uTree.init( global_tree_level, UMat, ctx, runtime );
  VTree vTree; vTree.init( global_tree_level, VMat, ctx, runtime );
  KTree kTree; kTree.init( matrix_level, UMat, VMat, DVec, ctx, runtime );
  VTree vTreep; vTreep.init( global_tree_level, VMatp, ctx, runtime );
  
  // data partition
  uTree.horizontal_partition( task_level, ctx, runtime );
  vTree.horizontal_partition( task_level, ctx, runtime );
  kTree.horizontal_partition( task_level, ctx, runtime );
  vTreep.horizontal_partition( task_level, ctx, runtime );
  
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
    LMatrix& V = vTreep.level_new(l);
    LMatrix& u = uTree.uMat_level_new(l);
    LMatrix& d = uTree.dMat_level_new(l);    

    // compute local results
    LogicalRegion VTu_ghost = task->regions[ghost_idx].region;
    LogicalRegion VTd_ghost = task->regions[ghost_idx+1].region;
    LogicalRegion VTu_local = create_local_region(VTu_ghost, ctx, runtime);
    LogicalRegion VTd_local = create_local_region(VTd_ghost, ctx, runtime);
    LMatrix VTu = create_legion_matrix(VTu_local, ctx, runtime);
    LMatrix VTd = create_legion_matrix(VTd_local, ctx, runtime);

    LMatrix::gemm('t', 'n', 1.0, V, u, 0.0, VTu, ctx, runtime );
    LMatrix::gemm('t', 'n', 1.0, V, d, 0.0, VTd, ctx, runtime );
    
    // copy
    CopyLauncher  cp_VTu;
    CopyLauncher  cp_VTd;
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
    LMatrix VTdp;
    LogicalRegion VTup_ghost = task->regions[ghost_idx+2].region;
    LogicalRegion VTdp_ghost = task->regions[ghost_idx+3].region;
    if (is_master_task(spmd_point, l, spmd_level)) {
      LMatrix VTu0 = create_legion_matrix(VTu_ghost, ctx, runtime);
      LMatrix VTd0 = create_legion_matrix(VTd_ghost, ctx, runtime);
      LMatrix VTu1 = create_legion_matrix(VTup_ghost, ctx, runtime);
      LMatrix VTd1 = create_legion_matrix(VTdp_ghost, ctx, runtime);
      args->reduction[l] = 
	runtime->advance_phase_barrier(ctx, args->reduction[l]);
      assert(args->node_solve[l]!=PhaseBarrier());
      LMatrix::node_solve( VTu0, VTu1, VTd0, VTd1,
			   args->reduction[l], args->node_solve[l],
			   ctx, runtime );

      VTdp = VTd1;
    }

    // copy data
    else {
      LogicalRegion VTdp_local = create_local_region(VTdp_ghost,ctx,runtime);
      VTdp = create_legion_matrix(VTdp_local, ctx, runtime); 
      args->node_solve[l] = 
	runtime->advance_phase_barrier(ctx, args->node_solve[l]);
      CopyLauncher cp_node_solve;
      cp_node_solve.add_copy_requirements
	(RegionRequirement(VTdp_ghost, READ_ONLY, EXCLUSIVE, VTdp_ghost),
	 RegionRequirement(VTdp_local, WRITE_DISCARD, EXCLUSIVE, VTdp_local));
      cp_node_solve.add_src_field(0, FIELDID_V);
      cp_node_solve.add_dst_field(0, FIELDID_V);
      cp_node_solve.add_wait_barrier(args->node_solve[l]);
      runtime->issue_copy_operation(ctx, cp_node_solve);
    }

    ghost_idx += 4;      
    
    // update: d -= u * VTd
    bool wait = (l==0 ? true : false);
    LMatrix::gemm_inplace('n', 'n', -1.0, u, VTdp, 1.0, d,
			  ctx, runtime, wait );
    std::cout<<"launched solver tasks at level: "<<l<<std::endl;
  }
  // check residule

  // clear resources
  uTree.clear(ctx, runtime);
  vTree.clear(ctx, runtime);
  kTree.clear(ctx, runtime);
}

void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, Runtime *runtime) {
 
  // machine configuration
  const int num_machines = 2;
  int num_cores_per_machine = 1;
 
  // HODLR configuration
  int rank0 = 100;
  int rank1 = 100;
  int leaf_size = 400;
  int matrix_level = 1;

  // right hand side
  const int nRhs = 1;

  // parse input arguments
  const InputArgs &command_args = Runtime::get_input_args();
  if (command_args.argc > 1) {
    for (int i = 1; i < command_args.argc; i++) {
      //if (!strcmp(command_args.argv[i],"-machine"))
	//num_machines = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-core"))
	num_cores_per_machine = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-rank0"))
	rank0 = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-rank1"))
	rank1 = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-leaf"))
	leaf_size = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-mtxlvl"))
	matrix_level = atoi(command_args.argv[++i]);
    }
  }
  int spmd_level = (int)log2(num_machines);
  int task_level = (int)log2(num_cores_per_machine);
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
	   <<"\noff-diagonal rank: "<<rank0<<", "<<rank1
	   <<"\nleaf size: "<<leaf_size
	   <<"\nmatrix level: "<<matrix_level
           <<"\n========================\n"
	   <<std::endl;

  assert(is_power_of_two(num_machines));
  assert(is_power_of_two(num_cores_per_machine));
  assert(rank0         > 0);
  assert(rank1         > 0);
  assert(leaf_size     > 0);
  assert(spmd_level<=MAX_TREE_LEVEL);
  
  // create phase barriers
  SPMDargs arg;
  arg.leaf_size = leaf_size;
  arg.rank0 = rank0;
  arg.rank1 = rank1;
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
  
  // create ghost regions: VTu0(r1 x r0), VTu1(r0 x r1) and VTd0(r1 x ...), VTd1(r0 x ...)
  Point<2> lo = make_point(0, 0);
  Point<2> hi0 = make_point(rank1-1, rank0-1);
  Point<2> hi1 = make_point(rank0-1, rank1-1);
  Rect<2>  rect0(lo, hi0);
  Rect<2>  rect1(lo, hi1);
  IndexSpace VTu0_is = runtime->create_index_space
    (ctx, Domain::from_rect<2>(rect0));
  IndexSpace VTu1_is = runtime->create_index_space
    (ctx, Domain::from_rect<2>(rect1));
  runtime->attach_name(VTu0_is, "VTu0_ghost_is");
  FieldSpace fs = runtime->create_field_space(ctx);
  runtime->attach_name(fs, "VTu_ghost_fs");
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double), FIELDID_V);
    runtime->attach_name(fs, FIELDID_V, "GHOST");
  }
  // go upward (leaf -> root), this order should be consistant with task
  // launches, so the right ghost region can be referred to.
  std::vector<std::vector<LogicalRegion> > VTu_all;
  std::vector<std::vector<LogicalRegion> > VTd_all;
  for (int l=spmd_level-1; l>=0; l--) {
    int num_ghosts = (int)pow(2, l);
    int num_shards_per_ghost = (int)pow(2, spmd_level-l);
    // create ghost regions
    std::vector<std::vector<LogicalRegion> > VTu_ghosts(num_ghosts);
    std::vector<std::vector<LogicalRegion> > VTd_ghosts(num_ghosts);
    for (int i=0; i<num_ghosts; i++) {
      // VTu0
      VTu_ghosts[i].push_back(runtime->create_logical_region(ctx, VTu0_is, fs));
      // VTu1
      VTu_ghosts[i].push_back(runtime->create_logical_region(ctx, VTu1_is, fs));
    }
    // create VTd regions
    Point<2> lo = make_point(0, 0);
    Point<2> hi0 = make_point(rank1-1, nRhs+l*rank0-1);
    Point<2> hi1 = make_point(rank0-1, nRhs+l*rank1-1);
    Rect<2>  rect0(lo, hi0);
    Rect<2>  rect1(lo, hi1);
    IndexSpace VTd0_is = runtime->create_index_space
      (ctx, Domain::from_rect<2>(rect0));
    IndexSpace VTd1_is = runtime->create_index_space
      (ctx, Domain::from_rect<2>(rect1));
    for (int i=0; i<num_ghosts; i++) {
      // VTd0
      VTd_ghosts[i].push_back(runtime->create_logical_region(ctx, VTd0_is, fs));
      // VTd1
      VTd_ghosts[i].push_back(runtime->create_logical_region(ctx, VTd1_is, fs));
    }
    VTu_all.insert(VTu_all.end(), VTu_ghosts.begin(), VTu_ghosts.end());
    VTd_all.insert(VTd_all.end(), VTd_ghosts.begin(), VTd_ghosts.end());
    // add region requirements
    for (int shard=0; shard<num_machines; shard++) {
      int idx    = shard / num_shards_per_ghost;
      int subidx = shard % num_shards_per_ghost / (num_shards_per_ghost/2);
      assert(subidx==0||subidx==1);

      // master task owns all ghost regions
      if (is_master_task(shard, l, spmd_level)) {
	assert(subidx==0);
	LogicalRegion VTu0 = VTu_ghosts[idx][0];
	LogicalRegion VTd0 = VTd_ghosts[idx][0];
	LogicalRegion VTu1 = VTu_ghosts[idx][1];
	LogicalRegion VTd1 = VTd_ghosts[idx][1];
	// reduce and then read
	RegionRequirement VTu0_req(VTu0,READ_WRITE,SIMULTANEOUS,VTu0);
	// read and then write
	RegionRequirement VTd0_req(VTd0,READ_WRITE,SIMULTANEOUS,VTd0);
	// read only
	RegionRequirement VTu1_req(VTu1,READ_ONLY,SIMULTANEOUS,VTu1);
	// read and then write
	RegionRequirement VTd1_req(VTd1,READ_WRITE,SIMULTANEOUS,VTd1);
	VTu0_req.add_field(FIELDID_V);
	VTd0_req.add_field(FIELDID_V);
	VTu1_req.add_field(FIELDID_V);
	VTd1_req.add_field(FIELDID_V);
	spmd_tasks[shard].add_region_requirement(VTu0_req);
	spmd_tasks[shard].add_region_requirement(VTd0_req);
	spmd_tasks[shard].add_region_requirement(VTu1_req);
	spmd_tasks[shard].add_region_requirement(VTd1_req);
      }

      // declare no access to ghost regions
      else {
	int subidxp = 1 - subidx;
	LogicalRegion VTu = VTu_ghosts[idx][subidx];
	LogicalRegion VTd = VTd_ghosts[idx][subidx];
	LogicalRegion VTup = VTu_ghosts[idx][subidxp];
	LogicalRegion VTdp = VTd_ghosts[idx][subidxp];
	// reduce
	RegionRequirement VTu_req(VTu,READ_WRITE,SIMULTANEOUS,VTu);
	// reduce and then read
	RegionRequirement VTd_req(VTd,READ_WRITE,SIMULTANEOUS,VTd);
	// nothing actually
	RegionRequirement VTup_req(VTup,READ_ONLY,SIMULTANEOUS,VTup);
	// read only
	RegionRequirement VTdp_req(VTdp,READ_ONLY,SIMULTANEOUS,VTdp);
	VTu_req.flags = NO_ACCESS_FLAG;
	VTd_req.flags = NO_ACCESS_FLAG;
	VTup_req.flags = NO_ACCESS_FLAG;
	VTdp_req.flags = NO_ACCESS_FLAG;
	VTu_req.add_field(FIELDID_V);
	VTd_req.add_field(FIELDID_V);
	VTup_req.add_field(FIELDID_V);
	VTdp_req.add_field(FIELDID_V);
	spmd_tasks[shard].add_region_requirement(VTu_req);
	spmd_tasks[shard].add_region_requirement(VTd_req);
	spmd_tasks[shard].add_region_requirement(VTup_req);
	spmd_tasks[shard].add_region_requirement(VTdp_req);
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

  // destroy regions
  for (unsigned i=0; i<VTu_all.size(); i++) {
    for (unsigned j=0; j<VTu_all[i].size(); j++) {
      runtime->destroy_logical_region(ctx, VTu_all[i][j]);
    }
  }
  for (unsigned i=0; i<VTd_all.size(); i++) {
    for (unsigned j=0; j<VTd_all[i].size(); j++) {
      runtime->destroy_logical_region(ctx, VTd_all[i][j]);
    }
  }
}

int main(int argc, char *argv[]) {
  // register top level task
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  Runtime::register_legion_task<top_level_task>(
    TOP_LEVEL_TASK_ID,   /* task id */
    Processor::LOC_PROC, /* cpu */
    true,  /* single */
    false, /* index  */
    AUTO_GENERATE_ID,
    TaskConfigOptions(false /*leaf task*/),
    "master-task"
  );
  Runtime::register_legion_task<spmd_fast_solver>(SPMD_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(false), "spmd");

  // register solver tasks
  register_solver_tasks();

  // start legion master task
  return Runtime::start(argc, argv);
}
