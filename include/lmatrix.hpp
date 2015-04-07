#ifndef _lmatrix_hpp
#define _lmatrix_hpp

#include <string>

#include "legion.h"
using namespace LegionRuntime::HighLevel;

#include "matrix.hpp"

// legion matrix
class LMatrix {
public:
  LMatrix();

  // for UTree::init_rhs()
  void init(const Vector& Rhs);

  // for UTree::rhs()
  Vector to_vector();

  // for VTree::copy()
  void create(const Matrix& VMat, Context, HighLevelRuntime*);

  // for KTree::partition()
  void create_dense_blocks
  (int, const Matrix& U, const Matrix& V, const Vector& D,
   Context, HighLevelRuntime*);
  
  // for VTree::partition() and
  // for this->create_dense_blocks()
  //  uniformly partition
  void partition(int level, Context, HighLevelRuntime*);

  // solve linear system
  void solve(LMatrix&, Context, HighLevelRuntime*, bool wait=true);

  void node_solve(LMatrix&, Context, HighLevelRuntime*, bool wait=true);
  
  // print the values on screen
  void display(const std::string&, Context, HighLevelRuntime*);

  // gemm reduction
  static void gemmRed
  (double, const LMatrix&, const LMatrix&,
   double, const LMatrix&, Context, HighLevelRuntime*, bool wait=true);
  
  // gemm broadcast
  static void gemmBro
  (double, const LMatrix&, const LMatrix&,
   double, const LMatrix&, Context, HighLevelRuntime*, bool wait=true);
  
private:

  //int mRows, mCols, mblock;
  int nProc;
  int nPart;
    
  IndexSpace       ispace;
  FieldSpace       fspace;
  Blockify<2>      blkify;
  Domain           domain;
  LogicalRegion    region;
  IndexPartition   ipart;
  //LogicalPartition lpart;
};

/*
// legion matrix
class LMatrix {
public:
  LMatrix();

  // get the number of partitions
  int num_partition();

  // get the logical partition
  LogicalPartition logical_partition();

  // get the coloar domain
  Domain color_domain();
    
  // initialize dense diagonal blocks
  void init
  (const int nProc, const Matrix& U, const Matrix& V, const Vector& D);

  // partition the data
  void partition(int level, Context, HighLevelRuntime*);

  // solve linear system
  void solve(LMatrix&, Context, HighLevelRuntime*, bool wait=true);

  void node_solve(LMatrix&, Context, HighLevelRuntime*, bool wait=true);
  
  // print the values on screen
  void display(const std::string&, Context, HighLevelRuntime*);

  // gemm reduction
  static void gemmRed
  (double, const LMatrix&, const LMatrix&,
   double, const LMatrix&, Context, HighLevelRuntime*, bool wait=true);
  
  // gemm broadcast
  static void gemmBro
  (double, const LMatrix&, const LMatrix&,
   double, const LMatrix&, Context, HighLevelRuntime*, bool wait=true);
  
private:

  // helper for solve() and node_solve()
  template <typename SolveType>
  void solve(LMatrix&, Context, HighLevelRuntime*, bool wait=true);
  
  //int mRows, mCols, mblock;
  int nProc;
  int nPart;
  
  Matrix U, V;
  Vector D;
  
  IndexSpace       ispace;
  FieldSpace       fspace;
  Blockify<2>      blkify;
  Domain           domain;
  LogicalRegion    region;
  IndexPartition   ipart;
  //LogicalPartition lpart;
};
*/

#endif
