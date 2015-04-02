#include <iostream>

// legion stuff
#include "legion.h"
using namespace LegionRuntime::HighLevel;

#include "matrix.hpp"  // for Matrix  class
#include "hmatrix.hpp" // for HMatrix class

enum {
  TOP_LEVEL_TASK_ID = 0,
};

void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {  

  // ======= Problem configuration =======
  // solve: A x = b where A = U * V' + D
  // =====================================
  int N = 1<<10;
  int r = 30;
  Vector b(N);
  Matrix U(N, r);
  Matrix V(N, r);
  Vector D(N);

  // ================================================
  // generate random matrices, which could
  //  potentially be done in parallel
  // ================================================
  // number of processes, or number of ranks as in MPI
  int nProc = 2;
  b.rand( nProc );
  U.rand( nProc );
  V.rand( nProc );
  D.rand( nProc );

  // ========================================================
  // fast solver for a simple matrix U * V' + D
  //  where the off-diagonal blocks are exactly low rank,
  //  so the solve should be accurate (with round-off errors)
  // ========================================================
  // number of cores on each machine
  int nCore = 8;
  // number of levels for the (balanced) binary tree
  int level = 4;
  // assume one leaf every core for now
  assert( pow(2,level) == nProc*nCore );
  HMatrix Ah( nProc, nCore, level );
  Ah.init( Rhs, U, V, D );
  Ah.solve();

  /*
  // direct solve
  Matrix A = U*V.transpose() + D;
  Vector x = A/Rhs;
  */
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

  // start legion master task
  return HighLevelRuntime::start(argc, argv);
}
