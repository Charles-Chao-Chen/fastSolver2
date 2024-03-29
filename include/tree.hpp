#ifndef _tree_hpp
#define _tree_hpp

#include <vector>

#include "lmatrix.hpp"

class UTree {
public:

  // init data
  void init(const Matrix&);
  
  void init(int, const Matrix&, Context ctx, Runtime *runtime);
  
  // initialize problem right hand side
  void init_rhs
  (const Vector&, Context ctx, Runtime *runtime);

  void init_rhs
  (const Matrix&, Context ctx, Runtime *runtime,
   bool wait=false);
  
  // return right hand side (overwritten by solution)
  Vector rhs();

  // create partition
  void partition
  (int level, Context ctx, Runtime *runtime);
  
  void horizontal_partition
  (int level, Context ctx, Runtime *runtime);

  // return the solution
  Matrix solution(Context ctx, Runtime *runtime);
  
  // return the legion matrix for one level
  LMatrix& uMat_level(int);
  LMatrix& dMat_level(int);
  LMatrix& uMat_level_new(int);
  LMatrix& dMat_level_new(int);

  // legion matrices at leaf level
  LMatrix& leaf();

  void clear(Context ctx, Runtime* runtime);
  
private:
  int mLevel;
  int rank;
  int nRhs;
  Matrix  UMat;

  // ----------------------
  // legion matrices below
  // ----------------------
  
  // one large region for all data
  LMatrix U; 

  // u and d matrices at all levels
  std::vector<LMatrix> uMat_vec;
  std::vector<LMatrix> dMat_vec;
};

class VTree {
public:

  // init data
  void init(const Matrix&);
  
  void init(int, const Matrix&, Context ctx, Runtime *runtime);

  // create partition
  void partition
  (int level, Context ctx, Runtime *runtime);

  void horizontal_partition
  (int level, Context ctx, Runtime *runtime);

  // return the legion matrix for one level
  LMatrix& leaf();
  LMatrix& level(int);
  LMatrix& level_new(int);

  void clear(Context ctx, Runtime* runtime);
  
private:
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
  void init(const Matrix& U, const Matrix& V, const Vector& D);

  void init(int, const Matrix& U, const Matrix& V, const Vector& D,
	    Context ctx, Runtime *runtime);

  // create partition
  void partition
  (int level, Context ctx, Runtime *runtime);

  void horizontal_partition
  (int level, Context ctx, Runtime *runtime);

  // wrapper for legion matrix solve
  // leaf solve task
  void solve(LMatrix&, LMatrix&, Context ctx, Runtime *runtime);
  
  void clear(Context ctx, Runtime* runtime);

private:
  int mLevel;
  Matrix UMat, VMat;
  Vector DVec;
  LMatrix K;
};

#endif
