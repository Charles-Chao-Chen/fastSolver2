#ifndef _hmatrix_hpp
#define _hmatrix_hpp

#include "legion.h"

// legion matrix
class LMatrix {
public:

  int rows();
  int cols();
  int rblock();

  // for U matrix
  void init(const int nProc, const Matrix& b, const Matrix& U,
	    Context, HighLevelRuntime*);

  // for V matrix
  void init(const int nProc, const Matrix& V,
	    Context, HighLevelRuntime*);

  // for K matrix
  void init
  (const int nProc, const Matrix& U, const Matrix& V, const Matrix& D,
   Context, HighLevelRuntime*);

  // print the values on screen
  void display(Context, HighLevelRuntime*);

private:
  int mRows, mCols, mblock;
  IndexSpace    ispace;
  FieldSpace    fspace;
  Blockify<2>   blkify;
  IndexPartition ipart;
  LogicalRegion  region;
};

#endif
