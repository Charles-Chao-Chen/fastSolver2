#ifndef _hmatrix_hpp
#define _hmatrix_hpp

#include <vector>
#include "matrix.hpp"  // for  Matrix class
#include "lmatrix.hpp" // for LMatrix class

// the hierarchical tree is balanced
class HMatrix {
public:
  HMatrix();

  HMatrix(int nProc, int level);

  // build U * V' + D, with the right hand side b
  void init
  (const Matrix& U, const Matrix& V, const Matrix& D,
   Context, HighLevelRuntime*);

  // fast solver
  Matrix solve(const Matrix& b, Context, HighLevelRuntime*);

  // destructor
  void destroy(Context, HighLevelRuntime*);
  
private:
  int     nProc;
  int     level;
  UTree   uTree;
  VTree   vTree;
  LMatrix dBlck; // dense (diagonal) blocks
};

#endif
