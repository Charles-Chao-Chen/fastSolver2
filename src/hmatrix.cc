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
  this->nPart = pow(2, level);
  
  // assume evenly distributed across machines
  assert( nPart % nProc == 0);
}

HMatrix::init
(const Matrix& U, const Matrix& V, const Matrix& D,
 Context ctx, HighLevelRuntime* runtime) {
  
  // sanity check
  assert( b.rows() == U.rows() == V.rows() == D.rows() );
  assert( U.cols() == V.cols() >  0 );
  assert( D.rows() == D.cols() );
  assert( b.rows() >  0 && b.cols() >  0 );

  // populate data
  uTree.init( nProc, U, ctx, runtime );
  vTree.init( nProc, V, ctx, runtime );
  dBlck.init( nProc, U, V, D, ctx, runtime );

  // data partition
  uTree.partition( level );
  vTree.partition( level );
  dBlck.partition( level );
    
#ifdef DEBUG
  U.display("U");
  V.display("V");
  K.display("K");
#endif
}

Matrix HMatrix::solve
(const Matrix& b, Context ctx, HighLevelRuntime* runtime) {

  // initialize the right hand side
  uTree.init_rhs(b);
  
  // leaf solve: U = dense \ U
  dBlck.solve( uTree.leaf(), ctx, runtime );
  
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
  
  for (int i=Level; i>0; i--) {

    LMatrix  V = vTree.level(i);
    LMatrix& u = uTree.level(i).uMat;
    LMatrix& d = uTree.level(i).dMat;
    
    // reduction operation
    LMatrix VTu; gemmRed( 1.0, V, u, 0.0, VTu, ctx, runtime );
    LMatrix VTd; gemmRed( 1.0, V, d, 0.0, VTd, ctx, runtime );

    // form and solve the small linear system
    LMatrix S = LMatrix::Identity( VTu.rows() + VTd.rows(), ctx, runtime);
    S.set_off_diagonal_blcks( VTu, ctx, runtime );
    S.solve( VTd.coarse(), ctx, runtime );
    
    // broadcast operation
    // d -= u * VTd
    gemmBro( -1.0, u, VTd, 1.0, d, ctx, runtime );
  }
}
