#ifndef _lmatrix_hpp
#define _lmatrix_hpp

#include <string>

#include "legion.h"
using namespace LegionRuntime::HighLevel;

#include "macros.hpp" // for WAIT_DEFAULT
#include "matrix.hpp"

// legion matrix
class LMatrix {
public:
  LMatrix();

  int rows() const;
  int cols() const;
  int num_partition() const;
  Domain color_domain() const;
  LogicalRegion logical_region() const;
  LogicalPartition logical_partition() const;
  
  // for UTree::init_rhs()
  void init
  (const Vector& Rhs, Context, HighLevelRuntime*,
   bool wait=WAIT_DEFAULT);

  // for UTree::rhs()
  Vector to_vector();

  // for VTree::copy()
  void create
  (const Matrix& VMat, Context, HighLevelRuntime*,
   bool wait=WAIT_DEFAULT);

  // for KTree::partition()
  void create_dense_blocks
  (int, const Matrix& U, const Matrix& V, const Vector& D,
   Context, HighLevelRuntime*, bool wait=WAIT_DEFAULT);

  // uniform partition
  // for VTree::partition() and
  // for this->create_dense_blocks()
  void partition(int level, Context, HighLevelRuntime*);

  // solve linear system
  // for KTree::solve()
  void solve
  (LMatrix&, Context, HighLevelRuntime*, bool wait=WAIT_DEFAULT);

  // solve node system
  // for HMatrix::solve()
  void node_solve
  (LMatrix&, Context, HighLevelRuntime*, bool wait=WAIT_DEFAULT);

  // print the values on screen
  // for debugging
  void display(const std::string&, Context, HighLevelRuntime*);

  // gemm reduction
  static void gemmRed
  (double, const LMatrix&, const LMatrix&,
   double, const LMatrix&, Context, HighLevelRuntime*,
   bool wait=WAIT_DEFAULT);
  
  // gemm broadcast
  static void gemmBro
  (double, const LMatrix&, const LMatrix&,
   double, const LMatrix&, Context, HighLevelRuntime*,
   bool wait=WAIT_DEFAULT);
  
private:

  // helper functions
  void coarse_partition();
  void fine_partition();

  template <typename T>
  void solve
  (LMatrix& b, Context ctx, HighLevelRuntime *runtime,
   bool wait=WAIT_DEFAULT);

  // private variables
  int mRows, mCols;

  int nProc;
  int nPart;
    
  IndexSpace       ispace;
  FieldSpace       fspace;
  Blockify<2>      blkify;
  Domain           domain;
  LogicalRegion    region;
  IndexPartition   ipart;
  LogicalPartition lpart;
};

#endif
