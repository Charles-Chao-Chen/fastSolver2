#include "ptr_matrix.hpp"

#include <assert.h>
#include <stdlib.h> // for srand48_r(), lrand48_r() and drand48_r()

PtrMatrix::PtrMatrix() : mRows(-1), mCols(-1), LeadD(-1), ptr(NULL) {}

PtrMatrix::PtrMatrix(int r, int c, int l, double *p)
  : mRows(r), mCols(c), LeadD(l), ptr(p) {}

// assume row major storage,
//  which is consistant with blas and lapack layout
double* PtrMatrix::operator()(int r, int c) {
  return &ptr[r+c*LeadD];
}

void PtrMatrix::rand(long seed) {
  struct drand48_data buffer;
  assert( srand48_r( seed, &buffer ) == 0 );
  for (int i=0; i<mRows; i++) {
    for (int j=0; j<mCols; j++) {
      assert( drand48_r(&buffer, (*this)(i,j) ) == 0 );
    }
  }
}

void PtrMatrix::display(const std::string& name) {
  std::cout << name << ":" << std::endl;
  for(int ri = 0; ri < mRows; ri++) {
    for(int ci = 0; ci < mCols; ci++) {
      double *value = (*this)(ri, ci);
      std::cout << *value << "\t";
    }
    std::cout << std::endl;
  }
}
