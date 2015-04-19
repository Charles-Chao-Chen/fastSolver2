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
void test_lmatrix_init(Context, HighLevelRuntime*);
void test_leaf_solve(Context, HighLevelRuntime*);
void test_gemm_reduce(Context, HighLevelRuntime*);
void test_gemm_broadcast(Context, HighLevelRuntime*);

void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {  

  test_vector();
  test_matrix();
  //test_lmatrix_init(ctx, runtime);
  //test_leaf_solve(ctx, runtime);
  test_gemm_reduce(ctx, runtime);
  test_gemm_broadcast(ctx, runtime);
  
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

  Vector no_entry(N, false);
  no_entry.rand(nPart);
  no_entry.display("no_entry");
  
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

  if (Matrix::constant<1>(n,n) * Matrix::constant<1>(n,n)
      != Matrix::constant<n>(n,n))
    Error("matrix multiply wrong");

  if (Matrix::constant<10>(n,n) * Vector::constant<1>(n).to_diag_matrix()
      != Matrix::constant<10>(n,n))
    Error("vector to diagonal matrix wrong");
    
  Matrix no_entry(m, n, false);
  no_entry.rand(nPart);
  no_entry.display("no_entry");
  
  std::cout << "Test for Matrix passed!" << std::endl;
}

void test_lmatrix_init(Context ctx, HighLevelRuntime *runtime) {
  
  int m = 16, n = 2;
  int nPart = 2;
  Matrix  mat0(m, n);
  mat0.rand(nPart);
  mat0.display("mat0");
  
  LMatrix lmat0;
  lmat0.create(m, n, ctx, runtime);
  lmat0.init_data(nPart, 0, mat0.cols(), mat0, ctx, runtime);
  lmat0.display("lmat0", ctx, runtime);

  Matrix U(m, n), V(m, n);
  Vector D(m);
  U.rand(nPart);
  V.rand(nPart);
  D.rand(nPart);
  //U.display("U");
  //V.display("V");
  //D.display("D");
  
  int level = 3;
  int nrow = D.rows();
  int nblk = pow(2, level-1);
  int ncol = D.rows() / nblk;
  LMatrix lmat;
  lmat.create(nrow, ncol, ctx, runtime);
  lmat.init_dense_blocks(nPart, nblk, U, V, D, ctx, runtime);

  /*
  Matrix A = (U * V.T()) + D.to_diag_matrix();    
  A.display("full matrix");
  lmat.display("diagonal blocks", ctx, runtime);
*/
  LMatrix lgUmat;
  int cols = 1+level*U.cols();
  lgUmat.create(U.rows(), cols, ctx, runtime);
  lgUmat.init_data(nPart, 1, cols, U, ctx, runtime);

  // right hand side
  Matrix Rhs(m, 1);
  Rhs.rand(nPart);
  lgUmat.init_data(nPart, 0, 1, Rhs, ctx, runtime);

  Rhs.display("rhs");
  U.display("U");
  lgUmat.display("UTree", ctx, runtime);
  
  std::cout << "Test for legion matrix initialization passed!" << std::endl;
}

void test_leaf_solve(Context ctx, HighLevelRuntime *runtime) {

  int m = 16, n = 2;
  int nProc = 4;
  int level = 3;
  Matrix VMat(m, n), UMat(m, n), Rhs(m, 1);
  VMat.rand(nProc);
  UMat.rand(nProc);
  Rhs.rand(nProc);

  Vector DVec(m);
  DVec.rand(nProc);
  int nrow = DVec.rows();
  int nblk = pow(2, level);
  int ncol = DVec.rows() / nblk;
  assert(nblk >= nProc);
  assert(ncol >= UMat.cols());
  LMatrix K( nrow, ncol, level, ctx, runtime );;
  LMatrix K_copy( nrow, ncol, level, ctx, runtime );;
  K.init_dense_blocks(nProc, nblk, UMat, VMat, DVec, ctx, runtime);
  K_copy.init_dense_blocks(nProc, nblk, UMat, VMat, DVec, ctx, runtime);
  
  LMatrix b(Rhs.rows(), 1, level, ctx, runtime);
  LMatrix b_copy(Rhs.rows(), 1, level, ctx, runtime);
  b.init_data(nProc, 0, 1, Rhs, ctx, runtime);
  b_copy.init_data(nProc, 0, 1, Rhs, ctx, runtime);

  // linear solve
  K.solve( b, ctx, runtime );
  
  LMatrix Ax(Rhs.rows(), 1, level, ctx, runtime);
  LMatrix::gemmRed('n', 'n', 1.0, K_copy, b, 0.0, Ax, ctx, runtime);
  //Ax.display("Ax", ctx, runtime);

  LMatrix r(Rhs.rows(), 1, level, ctx, runtime);
  
  // r = b - Ax
  LMatrix::add(1.0, b_copy, -1.0, Ax, r, ctx, runtime);
  r.display("residule", ctx, runtime);

  /*
  b_copy.display("rhs", ctx, runtime);
  K.display("K", ctx, runtime);
  b.display("sln", ctx, runtime);
  Ax.display("Ax", ctx, runtime);
  */
    
  /*
  // test LMatrix::add
  LMatrix U0, U1;
  U0.create(UMat.rows(), UMat.cols(), ctx, runtime);
  U0.init_data(nProc, 0, UMat.cols(), UMat, ctx, runtime);
  U0.partition(level, ctx, runtime);
  U1.create(UMat.rows(), UMat.cols(), ctx, runtime);
  U1.init_data(nProc, 0, UMat.cols(), UMat, ctx, runtime);
  U1.partition(level, ctx, runtime);
  LMatrix res;
  res.create(UMat.rows(), UMat.cols(), ctx, runtime);
  res.partition(level, ctx, runtime);
  LMatrix::add(1.0, U0, -1.0, U1, res, ctx, runtime);
  res.display("result", ctx, runtime);
  LMatrix::add(1.0, U0,  0.0, U1, res, ctx, runtime);
  res.display("result", ctx, runtime);
  LMatrix::add(1.0, U0,  1.0, U1, res, ctx, runtime);
  res.display("result", ctx, runtime);
*/
    
  std::cout << "Test for leave solve passed!" << std::endl;
}

void test_gemm_reduce(Context ctx, HighLevelRuntime *runtime) {
  int m=16, n=3;
  int nProc = 4;
  Matrix UMat(m, n), VMat(m, n);
  UMat.rand(nProc);
  VMat.rand(nProc);

  int level = 3;
  LMatrix U(m, n, level, ctx, runtime);
  LMatrix V(m, n, level, ctx, runtime);
  U.init_data(nProc, 0, n, UMat, ctx, runtime);
  V.init_data(nProc, 0, n, VMat, ctx, runtime);

  // V^T * U
  LMatrix W(n, n, 0, ctx, runtime);
  LMatrix::gemmRed('t', 'n', 1.0, V, U, 0.0, W, ctx, runtime);
  W.display("W", ctx, runtime);

  Matrix WMat = VMat.T() * UMat;
  WMat.display("Wmat");
  /*
  VMat.display("Vmat");
  UMat.display("Umat");
*/
  std::cout << "Test for gemm reduce passed!" << std::endl;
}

void test_gemm_broadcast(Context ctx, HighLevelRuntime *runtime) {
  int m=16, n=3;
  int nProc = 4;
  Matrix UMat(m, n), VMat(n, n);
  UMat.rand(nProc);
  VMat.rand(1);

  int level = 3;
  LMatrix U(m, n, level, ctx, runtime);
  U.init_data(nProc, 0, n, UMat, ctx, runtime);
  LMatrix V(n, n, 0, ctx, runtime);
  V.init_data(1, 0, n, VMat, ctx, runtime);

  Matrix WMat = UMat * VMat;
  //UMat.display("UMat");
  //VMat.display("VMat");
  WMat.display("WMat");

  LMatrix W(m, n, level, ctx, runtime);
  LMatrix::gemmBro('n', 'n', 1.0, U, V, 0.0, W, ctx, runtime);
  W.display("W", ctx, runtime);
}
