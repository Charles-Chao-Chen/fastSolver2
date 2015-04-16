#ifndef _tree_hpp
#define _tree_hpp

#include <vector>

#include "lmatrix.hpp"

class UTree {
public:

  // init data
  void init(int, const Matrix&);
  
  // initialize problem right hand side
  void init_rhs(const Vector&, Context ctx, HighLevelRuntime *runtime);

  // return right hand side (overwritten by solution)
  Vector rhs();
  
  // create partition
  void partition
  (int level, Context ctx, HighLevelRuntime *runtime);

  /*
  // return type for level()
  struct UDMat {
    LMatrix uMat;
    LMatrix dMat;
  };
  */
  // return the legion matrix for one level
  LMatrix& uMat_level(int);
  LMatrix& dMat_level(int);
  
  // UDMat& level(int);

  // legion matrices at leaf level
  LMatrix& leaf();
  
private:
  int nProc;
  int mLevel;
  int rank;
  int nRhs;
  Matrix  UMat;

  // legion matrices
  LMatrix U; // original data

  // partitions at different levels
  //LMatrix Rhs;
  std::vector<LMatrix> uMat_vec;
  std::vector<LMatrix> dMat_vec;
  
  //std::vector<UDMat> Ulevel;
};

class VTree {
public:

  // init data
  void init(int, const Matrix&);

  // create partition
  void partition
  (int level, Context ctx, HighLevelRuntime *runtime);

  // return the legion matrix for one level
  LMatrix& level(int);
  
private:
  int nProc;
  int mLevel;
  Matrix VMat;

  // for the simple case of U * V' + D,
  // partition is the same for all levels,
  // so only one partition is stored
  LMatrix V;
};

// Dense blocks only exist at the leaf level
//  and are used for leaf solve task
class KTree {
public:
  
  // init data
  void init(int, const Matrix& U, const Matrix& V, const Vector& D);

  // create partition
  void partition
  (int level, Context ctx, HighLevelRuntime *runtime);

  // wrapper for legion matrix solve
  // leaf solve task
  void solve(LMatrix&, Context ctx, HighLevelRuntime *runtime);
  
private:
  int nProc;
  int mLevel;
  Matrix UMat, VMat;
  Vector DVec;
  LMatrix K;
};

#endif
