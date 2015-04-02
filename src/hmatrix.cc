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
(const Matrix& b, const Matrix& U, const Matrix& V, const Matrix& D) {
  // sanity check
  assert( b.rows() == U.rows() == V.rows() == D.rows() );
  assert( U.cols() == V.cols() >  0 );
  assert( D.rows() == D.cols() );
  assert( b.rows() >  0 && b.cols() >  0 );

  U.init( nProc, b, U );
  V.init( nProc, V );
  K.init( nProc, U, V, D );

#ifdef DEBUG
  U.display("U");
  V.display("V");
  K.display("K");
#endif
}

HMatrix::solve(Context ctx, HighLevelRuntime* runtime) {

  // leaf solve: U = K \ U
  K.solve(U.leaf(), ctx, runtime);
  
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

    
    
    GEMM_Reduce(V[i])
    NodeSolve();
    GEMM_Broadcast();
  }
}
