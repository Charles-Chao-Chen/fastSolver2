#ifndef _hmatrix_hpp
#define _hmatrix_hpp

#include "matrix.hpp" // for  Matrix class
#include "tree.hpp"   // for UTree, VTree and KTree

// the hierarchical tree is balanced
class HMatrix {
public:
  HMatrix();

  HMatrix(int nProc, int level);

  // build U * V' + D
  void init
  (const Matrix& U, const Matrix& V, const Vector& D,
   Context, HighLevelRuntime*);

  // fast solver
  Vector solve(const Matrix& b, Context, HighLevelRuntime*);

  // destructor
  void destroy(Context, HighLevelRuntime*);
  
private:

  // level=0 is a dense matrix
  // level=1 means the two off-diagonal blocks are low-rank
  int   nProc;
  int   level;
  UTree uTree;
  VTree vTree;
  KTree kTree;
};

#endif
