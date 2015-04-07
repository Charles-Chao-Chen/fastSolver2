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
  Vector solve(const Vector& b, Context, HighLevelRuntime*);

  // destructor
  void destroy(Context, HighLevelRuntime*);
  
private:
  int   nProc;
  int   level;
  UTree uTree;
  VTree vTree;
  KTree kTree;
};

#endif
