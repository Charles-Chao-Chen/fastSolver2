#ifndef _hmatrix_hpp
#define _hmatrix_hpp

#include <string>
#include "legion.h"

#include "matrix.hpp"

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
  
  /*
  int rows();
  int cols();
  int rblock();
  */
  /*
  // for U matrix
  void init(const int nProc, const Matrix& b, const Matrix& U,
	    Context, HighLevelRuntime*);

  // for V matrix
  void init(const int nProc, const Matrix& V,
	    Context, HighLevelRuntime*);
  */
  
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

private:
  //int mRows, mCols, mblock;
  int nProc;
  int nPart;
  
  Matrix U, V;
  Vector D;
  
  IndexSpace       ispace;
  FieldSpace       fspace;
  Blockify<2>      blkify;
  Domain           color_domain;
  LogicalRegion    region;
  IndexPartition   ipart;
  //LogicalPartition lpart;
};

#endif
