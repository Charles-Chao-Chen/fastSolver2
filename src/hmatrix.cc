#include "hmatrix.hpp"

HMatrix::HMatrix() {}

HMatrix::HMatrix(int nProc_) : nProc(nProc_), nPart(2*nProc_) {}

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

  // U = K.solve(U)
  LeafSolve(K, U, ctx, runtime);
  
  // upward pass
  for (int i=Level; i>0; i--) {
    // 
    GEMM_Reduce()
    NodeSolve();
    GEMM_Broadcast();
  }
}
