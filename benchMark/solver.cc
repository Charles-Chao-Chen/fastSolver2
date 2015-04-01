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

  Matrix b;
  Matrix U, V, D;

  // number of processes, or number of ranks as in MPI
  int nProc = 2;
  b.rand( nProc );
  U.rand( nProc );
  V.rand( nProc );
  D.rand( nProc );
  
  // fast solver
  HMatrix Ah( nProc );
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
