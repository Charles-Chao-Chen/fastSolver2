#ifndef _hmatrix_hpp
#define _hmatrix_hpp

#include <vector>
#include "lmatrix.hpp"

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
  void solve(const Matrix& b, Context, HighLevelRuntime*);
  
private:
  int nProc;
  //int nPart;
  int level;
  LMatrix U;
  std::vector<LMatrix> V;
  LMatrix K;
};

#endif
