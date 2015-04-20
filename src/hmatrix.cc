#include <math.h> // for pow()

#include "hmatrix.hpp"

HMatrix::HMatrix() {}

HMatrix::HMatrix(int nProc_, int level_)
  : nProc(nProc_), level(level_) {

  // ================================================
  // the first step is to have the same number of
  //  partitions as the number of leaves.
  // ================================================
  // number of partitions
  // if level=1, the number of partition is 2 for V0 and V1
  int nPart = pow(2, level);
  
  // assume evenly distributed across machines
  assert( nPart % nProc == 0);
}

void HMatrix::init
(const Matrix& U, const Matrix& V, const Vector& D,
 Context ctx, HighLevelRuntime* runtime) {
  
  // sanity check
  assert( U.rows() == V.rows() );
  assert( U.rows() == D.rows() );
  assert( U.cols() == V.cols() );
  assert( U.cols()  > 0 );

  // populate data
  uTree.init( nProc, U);
  vTree.init( nProc, V );
  kTree.init( nProc, U, V, D );

  // data partition
  uTree.partition( level, ctx, runtime );
  vTree.partition( level, ctx, runtime );
  kTree.partition( level, ctx, runtime );
    
#ifdef DEBUG
  uTree.display("U");
  vTree.display("V");
  kTree.display("K");
#endif
}

Vector HMatrix::solve
(const Matrix& b, Context ctx, HighLevelRuntime* runtime) {

  // check input
  assert( b.rows() > 0 );
  assert( b.cols() == 1 ); // only support a single right hand side now
  
  // initialize the right hand side
  uTree.init_rhs(b, ctx, runtime);
  
  // leaf solve: U = dense \ U
  kTree.solve( uTree.leaf(), ctx, runtime );
  
  // upward pass:
  // --             --  --    --     --      --
  // |  I     V1'*u1 |  | eta0 |     | V1'*d1 |
  // |               |  |      |  =  |        |
  // | V0'*u0   I    |  | eta1 |     | V0'*d0 |
  // --             --  --    --     --      --
  //
  // -    -   --            --
  // | x0 |   | d0 - u0*eta0 |
  // |    | = |              |
  // | x1 |   | d1 - u1*eta1 |
  // -    -   --            --
  
  for (int i=level; i>0; i--) {

    LMatrix& V = vTree.level(i);
    LMatrix& u = uTree.uMat_level(i);
    LMatrix& d = uTree.dMat_level(i);
    
    // reduction operation
    LMatrix VTu(V.cols(), u.cols(), i, ctx, runtime);
    LMatrix VTd(V.cols(), d.cols(), i, ctx, runtime);
    LMatrix::gemmRed('t', 'n', 1.0, V, u, 0.0, VTu, ctx, runtime );
    LMatrix::gemmRed('t', 'n', 1.0, V, d, 0.0, VTd, ctx, runtime );
    
    // form and solve the small linear system
    VTu.node_solve( VTd, ctx, runtime );
      
    // broadcast operation
    // d -= u * VTd
    LMatrix::gemmBro('n', 'n', -1.0, u, VTd, 1.0, d, ctx, runtime );
  }
  
  return uTree.rhs();
}
