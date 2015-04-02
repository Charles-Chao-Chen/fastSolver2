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

  this->U.init( nProc, U, ctx, runtime );

  for (int i=0; i<level; i++) {
    LMatrix temp;
    temp.init( nProc, nPart, V, ctx, runtime );
    this->V.push_back();
  }
  
  this->K.init( nProc, nPart, U, V, D, ctx, runtime );

  U.tree_partition( level );
  for (int i=0; i<level; i++) {this->V[i].partition(i);}
  K.partition( level );
    
#ifdef DEBUG
  U.display("U");
  V.display("V");
  K.display("K");
#endif
}

HMatrix::solve(const Matrix& b, Context ctx, HighLevelRuntime* runtime) {

  U.init_rhs(b);
  
  // leaf solve: U = K \ U
  K.solve( U.leaf(), ctx, runtime );
  
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

    // reduction operation
    LMatrix VTu = V[i].T() * U.part_u[i];
    LMatrix VTd = V[i].T() * U.part_d[i];

    // form and solve the small linear system
    VTu.form_square().solve( VTd.coarse(), ctx, runtime );
    
    // broadcast operation
    U.part_d[i] -= U.part_u[i] * VTd;
  }
}
