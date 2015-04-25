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
void test_node_solve(Context, HighLevelRuntime*);
void test_one_level(Context, HighLevelRuntime*);

void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {  

  test_vector();
  test_matrix();
  //test_lmatrix_init(ctx, runtime);
  //test_leaf_solve(ctx, runtime);  
  //test_gemm_reduce(ctx, runtime);
  //test_gemm_broadcast(ctx, runtime);
  //test_node_solve(ctx, runtime);
  test_one_level(ctx, runtime);

    
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
  int nPart = 4;
  Matrix mat0(m, n);
  mat0.rand(nPart);

  int nlevel = 2;
  assert(nPart == pow(2,nlevel));
  LMatrix lmat0(m, n, nlevel, ctx, runtime);
  lmat0.init_data(mat0, ctx, runtime);
  lmat0.display("lmat0", ctx, runtime);
  Matrix check0 = lmat0.to_matrix(ctx, runtime) - mat0;
  check0.display("init data residule");  

  Matrix UMat(m, n);
  UMat.rand(nPart);
  int nRhs = 1;
  int cols = nRhs+nlevel*n;
  LMatrix lgUmat(m, cols, nlevel, ctx, runtime);
  
  // right hand side
  Matrix Rhs(m, nRhs);
  Rhs.rand(nPart);
  lgUmat.init_data(Rhs, ctx, runtime);
  lgUmat.init_data(nRhs, cols, UMat, ctx, runtime);
  //Rhs.display("Rhs");
  //UMat.display("UMat");
  lgUmat.display("UTree", ctx, runtime);
  Matrix check1 = lgUmat.to_matrix(0, nRhs, ctx, runtime) - Rhs;
  check1.display("rhs residule");  
  Matrix check2 = lgUmat.to_matrix(nRhs, nRhs+n, ctx, runtime) - UMat;
  check2.display("Umat residule");  
  
  Matrix U(m, n), V(m, n);
  Vector D(m);
  U.rand(nPart);
  V.rand(nPart);
  D.rand(nPart);
  
  int nrow = D.rows();
  int nblk = pow(2, nlevel);
  int ncol = D.rows() / nblk;
  LMatrix lmat(nrow, ncol, nlevel, ctx, runtime);
  lmat.init_dense_blocks(U, V, D, ctx, runtime);
  lmat.display("dense blocks", ctx, runtime);
  Matrix KMat = (U * V.T()) + D.to_diag_matrix();
  Matrix check3 = lmat.to_matrix(0,m/nPart,0,m/nPart,ctx,runtime)
    - KMat.block(0,m/nPart,0,m/nPart);
  check3.display("dense block residule");
  if (check0.norm()<1.0e-13 && check1.norm()<1.0e-13 &&
      check2.norm()<1.0e-13 && check3.norm()<1.0e-13 ) {
    std::cout << "Test for legion matrix initialization passed!"
	      << std::endl;
  }
}

void test_leaf_solve(Context ctx, HighLevelRuntime *runtime) {

  int m = 16, n = 2;
  int nProc = 4;
  int level = 2;
  assert(nProc==pow(2, level));
  Matrix VMat(m, n), UMat(m, n), Rhs(m, 1);
  VMat.rand(nProc);
  UMat.rand(nProc);
  Rhs.rand(nProc);

  Vector DVec(m);
  DVec.rand(nProc);
  int nrow = DVec.rows();
  int nblk = pow(2, level);
  int ncol = DVec.rows() / nblk;
  LMatrix K( nrow, ncol, level, ctx, runtime );
  LMatrix K_copy( nrow, ncol, level, ctx, runtime );
  K.init_dense_blocks(UMat, VMat, DVec, ctx, runtime);
  K_copy.init_dense_blocks(UMat, VMat, DVec, ctx, runtime);
  
  LMatrix b(Rhs.rows(), 1, level, ctx, runtime);
  LMatrix b_copy(Rhs.rows(), 1, level, ctx, runtime);
  b.init_data(Rhs, ctx, runtime);
  b_copy.init_data(Rhs, ctx, runtime);

  // linear solve
  K.solve( b, ctx, runtime );
  
  LMatrix Ax(Rhs.rows(), 1, level, ctx, runtime);
  LMatrix::gemmRed('n', 'n', 1.0, K_copy, b, 0.0, Ax, ctx, runtime);
  LMatrix r(Rhs.rows(), 1, level, ctx, runtime);
  LMatrix::add(1.0, b_copy, -1.0, Ax, r, ctx, runtime);
  r.display("residule", ctx, runtime);
  Matrix res = r.to_matrix(ctx, runtime);
  if (res.norm()<1.0e-13) {
    std::cout << "Test for leave solve passed!" << std::endl;
  }
}

void test_gemm_reduce(Context ctx, HighLevelRuntime *runtime) {
  int m=16, n=3;
  int nProc = 4;
  int level = 2;
  assert(nProc == pow(2,level));
  Matrix UMat(m, n), VMat(m, n);
  UMat.rand(nProc);
  VMat.rand(nProc);

  Matrix WMat0 = VMat.block(0,m/2,0,n).T() * UMat.block(0,m/2,0,n);
  Matrix WMat1 = VMat.block(m/2,m,0,n).T() * UMat.block(m/2,m,0,n);

  LMatrix U(m, n, level, ctx, runtime);
  LMatrix V(m, n, level, ctx, runtime);
  U.init_data(UMat, ctx, runtime);
  V.init_data(VMat, ctx, runtime);

  // V^T * U
  LMatrix W(2*n, n, 1, ctx, runtime);
  LMatrix::gemmRed('t', 'n', 1.0, V, U, 0.0, W, ctx, runtime);
  W.display("W", ctx, runtime);
  Matrix check0 = W.to_matrix(0,n,0,n,ctx,runtime) - WMat0;
  Matrix check1 = W.to_matrix(n,2*n,0,n,ctx,runtime) - WMat1;
  check0.display("gemm residual");
  check1.display("gemm residual");
  if (check0.norm()<1.0e-13 && check1.norm()<1.0e-13) {
    std::cout << "Test for gemm reduce passed!" << std::endl;
  }
}

void test_gemm_broadcast(Context ctx, HighLevelRuntime *runtime) {
  int m=16, n=3;
  int nProc = 8;
  Matrix UMat(m, n), VMat(2*n, n);
  UMat.rand(nProc);
  VMat.rand(2);
  
  Matrix WMat0 = UMat.block(0,8,0,n) * VMat.block(0,n,0,n);
  Matrix WMat1 = UMat.block(8,16,0,n) * VMat.block(n,2*n,0,n);
  
  int level = 3;
  assert(nProc==pow(2,level));
  LMatrix U(m, n, level, ctx, runtime);
  U.init_data(UMat, ctx, runtime);
  LMatrix V(2*n, n, 1, ctx, runtime);
  V.init_data(VMat, ctx, runtime);
  LMatrix W(m, n, level, ctx, runtime);
  LMatrix::gemmBro('n', 'n', 1.0, U, V, 0.0, W, ctx, runtime);
  W.display("W", ctx, runtime);
  Matrix r0 = W.to_matrix(0,m/2,0,n,ctx,runtime) - WMat0;
  Matrix r1 = W.to_matrix(m/2,m,0,n,ctx,runtime) - WMat1;
  r0.display("gemm residual");
  r1.display("gemm residual");
  if (r0.norm()<1.0e-13 && r1.norm()<1.0e-13) {
    std::cout << "Test for gemm broadcast passed!" << std::endl;
  }
}

void test_node_solve(Context ctx, HighLevelRuntime *runtime) {
  int level = 2;
  int r = 3;
  int nProc = 2;
  Matrix VuMat(level*2*r, r), VdMat(level*2*r, 1);
  VuMat.rand(nProc);
  VdMat.rand(nProc);

  Matrix sln[2];
  for (int l=0; l<level; l++) {
    std::cout << "level: " << l << std::endl;
    int ofs = l*2*r;
    Matrix S = Matrix::identity(2*r);
    for (int i=0; i<r; i++) {
      for (int j=0; j<r; j++) {
	S(r+i, j) = VuMat(ofs+i, j);
	S(i, r+j) = VuMat(ofs+r+i, j);
      }
    }    
    //S.display("S");
    Matrix rhs(2*r, 1);
    for (int i=0; i<r; i++) {
      rhs(i, 0) = VdMat(ofs+r+i, 0);
      rhs(r+i, 0) = VdMat(ofs+i, 0);
    }
    //rhs.display("rhs");
    Matrix rhs_copy = rhs;
    Matrix A = S;
    A.solve(rhs);
    sln[l] = rhs;
    //rhs.display("sln");
  
    Matrix b = rhs_copy - S*rhs;
    //b.display("residule");
  }

  int nlevel = 1;
  assert(nProc==pow(2,nlevel));
  LMatrix VTu(level*2*r, r, nlevel, ctx, runtime);
  LMatrix VTd(level*2*r, 1, nlevel, ctx, runtime);
  VTu.init_data(VuMat, ctx, runtime);
  VTd.init_data(VdMat, ctx, runtime);
  VTu.node_solve(VTd, ctx, runtime);
  VTd.display("VTd", ctx, runtime);

  Matrix r0 = VTd.to_matrix(0,2*r,0,1,ctx,runtime) - sln[0];
  Matrix r1 = VTd.to_matrix(2*r,4*r,0,1,ctx,runtime) - sln[1];  
  r0.display("node solve residual");
  r1.display("node solve residual");
  if (r0.norm()<1.0e-13&&r1.norm()<1.0e-13)
    std::cout << "Test for node solve passed!" << std::endl;
}

void test_one_level(Context ctx, HighLevelRuntime *runtime) {

  int m = 8, n = 2;
  int nProc = 2;
  int level = 1;
  assert(nProc==pow(2,level));
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
  LMatrix K( nrow, ncol, level, ctx, runtime );

  /*
  LMatrix b(m, 1, level, ctx, runtime);
  LMatrix U(m, n, level, ctx, runtime);
  b.init_data(Rhs, ctx, runtime);
  U.init_data(UMat, ctx, runtime);
  
  // linear solve
  K.init_dense_blocks(UMat, VMat, DVec, ctx, runtime);
  K.solve( U, ctx, runtime );
  U.display("sln", ctx, runtime);
  K.init_dense_blocks(UMat, VMat, DVec, ctx, runtime);
  K.solve( b, ctx, runtime );
  b.display("sln", ctx, runtime);
  */
    
  UTree uTree;
  VTree vTree;
  KTree kTree;
  
  // init trees
  uTree.init( nProc, UMat);
  vTree.init( nProc, VMat );
  kTree.init( nProc, UMat, VMat, DVec );

  // data partition
  uTree.partition( level, ctx, runtime );
  vTree.partition( level, ctx, runtime );
  kTree.partition( level, ctx, runtime );

  // init rhs
  uTree.init_rhs(Rhs, ctx, runtime);
  
  // leaf solve: U = dense \ U
  kTree.solve( uTree.leaf(), ctx, runtime );  
  uTree.leaf().display("leaf solve", ctx, runtime);


  for (int i=level; i>0; i--) {

    LMatrix& V = vTree.level(i);
    LMatrix& u = uTree.uMat_level(i);
    LMatrix& d = uTree.dMat_level(i);

    //u.display("u", ctx, runtime);
    //d.display("d", ctx, runtime);
    

    // reduction operation
    int rows = pow(2, i)*V.cols();
    LMatrix VTu(rows, u.cols(), i, ctx, runtime);
    LMatrix VTd(rows, d.cols(), i, ctx, runtime);

    /*      
    LMatrix::gemmRed('t', 'n', 1.0, V, u, 0.0, VTu, ctx, runtime );

    VTu.display("VTu", ctx, runtime);
    Matrix vtu0 = VMat.row_block(0,4).T()*UMat.row_block(0,4);
    Matrix vtu1 = VMat.row_block(4,8).T()*UMat.row_block(4,8);
    VMat.display("VMat");
    UMat.display("UMat");
    vtu0.display("vtu0");
    vtu1.display("vtu1");
*/	

    //LMatrix::gemmRed('t', 'n', 1.0, V, d, 0.0, VTd, ctx, runtime );
    
    // form and solve the small linear system
    //VTu.node_solve( VTd, ctx, runtime );
      
    // broadcast operation
    // d -= u * VTd
    //LMatrix::gemmBro('n', 'n', -1.0, u, VTd, 1.0, d, ctx, runtime );

  }
  
  /*
  Matrix x = uTree.solution(ctx, runtime);

  // compute residule
  Matrix err = Rhs - ( UMat * (VMat.T() * x) + DVec.multiply(x) );
  err.display("err");
  std::cout << "Residual: " << err.norm() << std::endl;
*/
}
