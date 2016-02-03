#include <iostream>
#include <math.h>

// legion stuff
#include "legion.h"
using namespace LegionRuntime::HighLevel;

#include "matrix.hpp"  // for Matrix  class
#include "hmatrix.hpp" // for HMatrix class

enum {
  TOP_LEVEL_TASK_ID = 0,
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
  Matrix VMat(base, treelvl, n, has_entry); VMat.rand();
  Matrix UMat(base, treelvl, n, has_entry); UMat.rand();
  Matrix Rhs(base, treelvl, 1, has_entry);  Rhs.rand();
  Vector DVec(base, treelvl, has_entry);    DVec.rand(1e3);

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
  // fast solver for a simple matrix U * V' + D
  //  where the off-diagonal blocks are exactly low rank,
  //  so the solve should be accurate (with round-off errors)
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
      if (tracing && it==0 && i==1) {
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
  
  int rank = 100;
  int matrixlvl = 3;
  int tasklvl = 3;
  int niter = 1;
  bool tracing = false;
  const InputArgs &command_args = HighLevelRuntime::get_input_args();
  if (command_args.argc > 1) {
    for (int i = 1; i < command_args.argc; i++) {
      if (!strcmp(command_args.argv[i],"-rank"))
	rank = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-matrixlvl"))
	matrixlvl = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-tasklvl"))
	tasklvl = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-niter"))
	niter = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-tracing"))
	tracing = true;
    }
    assert(niter     > 0);
    assert(rank      > 0);
    assert(tasklvl   > 0);
    assert(matrixlvl >= tasklvl);
  }
  std::cout<<"Running fast solver..."
	   <<"\noff-diagonal rank: "<<rank
	   <<"\ntask-tree level: "<<tasklvl
	   <<"\nmatrix level: "<<matrixlvl
	   <<"\niteration number: "<<niter
	   <<"\nlegion tracing: "<<std::boolalpha<<tracing
	   <<std::endl;

  launch_solver_tasks(rank,matrixlvl,tasklvl,niter,tracing,ctx,runtime);
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

