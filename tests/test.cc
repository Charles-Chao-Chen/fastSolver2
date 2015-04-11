#include <iostream>
#include <math.h> // for fabs()

// legion stuff
#include "legion.h"
using namespace LegionRuntime::HighLevel;

#include "matrix.hpp"  // for Matrix  class
#include "hmatrix.hpp" // for HMatrix class

enum {
  TOP_LEVEL_TASK_ID = 0,
};

void test_vector();
void test_matrix();
void test_legion_matrix(Context, HighLevelRuntime*);

void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {  

  test_vector();
  test_matrix();
  test_legion_matrix(ctx, runtime);
  
  /*
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
  // number of levels for the (balanced) binary tree
  int level = 2;
  HMatrix Ah( nProc, level );
  Ah.init( U, V, D, ctx, runtime );
  Vector x = Ah.solve( b, ctx, runtime );

  // check solution
  Vector err = b - ( U * (V.T() * x) + D.multiply(x) );
  std::cout << "Residual: " << err.norm() << std::endl;
  */
}

int main(int argc, char *argv[]) {
  // register top level task
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
							 Processor::LOC_PROC, true/*single*/, false/*index*/);


  register_solver_tasks();
  
    
  /*
  HighLevelRuntime::register_single_task<top_level_task>(TOP_LEVEL_TASK_ID, Processor::LOC_PROC, false, "top_level_task");


  HighLevelRuntime::register_legion_task<top_level_task>(
    TOP_LEVEL_TASK_ID,
    Processor::LOC_PROC,
    true,
    false,
    AUTO_GENERATE_ID,
    TaskConfigOptions(false),
    "master-task
  );
*/
    
  // start legion master task
  return HighLevelRuntime::start(argc, argv);
}

void test_vector() {

  // test vector operations
  Vector vec1; // empty constructor

  int N = 16;
  Vector vec2(N); // build vector
  if (vec2.rows() != N)
    Error("inconsistant size");

  vec2 = Vector::constant<1>(N); // all one's
  if (fabs(vec2.norm() - 4) > 1e-10)
    Error("wrong 2-norm");
  
  int nPart = 4;
  vec2.rand(nPart); // random entries
  if (vec2.num_partition() != nPart)
    Error("wrong paritition number");
  //vec2.display("random vector");

  if (vec2+vec2 != 2*vec2)
    Error("+ or * wrong");

  if (vec2-vec2 != Vector::constant<0>(N))
    Error("- wrong");

  if (Vector::constant<3>(N).multiply(Vector::constant<5>(N))
      != Vector::constant<15>(N))
    Error("entry-wise muliply");

  std::cout << "Test for Vector passed!" << std::endl;
}

void test_matrix() {

  Matrix mat0; // empty constructor

  const int m = 16, n = 4;
  Matrix mat1(m, n);
  if (mat1.rows() != m || mat1.cols() != n)
    Error("wrong matrix size");

  int nPart = 4;
  mat1.rand(nPart);
  if (mat1.num_partition() != nPart)
    Error("wrong partition number");
  //mat1.display("random matrix");

  if (mat1+mat1 != 2*mat1)
    Error("+ or * wrong");

  if (mat1-mat1 != Matrix::constant<0>(m, n))
    Error("- wrong");

  if (Matrix::constant<1>(m,n) * Vector::constant<1>(n)
      != Vector::constant<n>(m))
    Error("mat-vec multiply wrong");

  if (Matrix::constant<17>(m,n).T() != Matrix::constant<17>(n,m))
    Error("matrix transpose wrong");

  if (Matrix::constant<1>(m,n).T() * Vector::constant<1>(m)
      != Vector::constant<m>(n))
    Error("transpose multiply wrong");
  
  std::cout << "Test for Matrix passed!" << std::endl;
}

void test_legion_matrix(Context ctx, HighLevelRuntime *runtime) {
  
  int m = 16, n = 2;
  int nPart = 4;
  Matrix  mat0(m, n);
  mat0.rand(nPart);
  mat0.display("mat0");
  
  LMatrix lmat0;
  lmat0.create(m, n, ctx, runtime);
  lmat0.init_data(nPart, mat0, ctx, runtime);
  lmat0.display("lmat0", ctx, runtime);
}
