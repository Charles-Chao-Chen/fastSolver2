#ifndef _hmatrix_hpp
#define _hmatrix_hpp

#include "lmatrix.hpp"

// the hierarchical tree is balanced
class HMatrix {
public:
  HMatrix();

  HMatrix(int nProc);
  
  // build U * V' + D, with the right hand side b
  void init
  (const Matrix& b, const Matrix& U, const Matrix& V, const Matrix& D,
   Context, HighLevelRuntime*);

  // fast solver
  void solve(Context, HighLevelRuntime*);
  
private:
  int nProc;
  int nPart;
  LMatrix U;
  LMatrix V;
  LMatrix K;
};

#endif
