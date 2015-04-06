#ifndef _tree_hpp
#define _tree_hpp

class UTree {
public:

  // constructor
  UTree(int, const Matrix&);

  // initialize problem right hand side
  void init_rhs(const Matrix&);

  // create partition
  void partition
  (int level, Context ctx, HighLevelRuntime *runtime);

  // return type for level()
  struct UDMat {
    LMatrix uMat;
    LMatrix dMat;
  };

  // return the legion matrix for one level
  UDMat level(int) const;

  // return the legion matrix at leaf level
  LMatrix leaf(int) const;
  
private:
  int nProc;
  int level;
  Matrix U;
  std::vector<UDMat> Upart;
};

class VTree {
public:

  // constructor
  VTree(int, const Matrix&);

  // create partition
  void partition
  (int level, Context ctx, HighLevelRuntime *runtime);

  // return the legion matrix for one level
  LMatrix level(int) const;
  
private:
  int nProc;
  int level;
  Matrix  V;

  // for the simple case of U * V' + D,
  // partition is the same for all levels,
  // so only one partition is stored
  LMatrix VPart;
};

// Dense blocks only exist at the leaf level
//  and are used for leaf solve task
class KTree : public HTree {
public:
  
  // constructor
  KTree(int, const Matrix& U, const Matrix& V, const Matrix& D);

  // create partition
  void partition
  (int level, Context ctx, HighLevelRuntime *runtime);

  // wrapper for leaf solve task
  void solve(LMatrix&, Context ctx, HighLevelRuntime *runtime);
  
private:
  int nProc;
  int level;
  Matrix  U, V, D;
  LMatrix KPart;
};

#endif
