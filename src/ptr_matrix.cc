#include "ptr_matrix.hpp"

#include <assert.h>
#include <stdlib.h> // for srand48_r(), lrand48_r() and drand48_r()

PtrMatrix::PtrMatrix()
  : mRows(-1), mCols(-1), leadD(-1), ptr(NULL),
    has_memory(false), trans('n') {}

PtrMatrix::PtrMatrix(int r, int c)
  : mRows(r), mCols(c), leadD(r),
    has_memory(true), trans('n') {
  ptr = new double[mRows*mCols];
}

PtrMatrix::~PtrMatrix() {
  if (has_memory)
    delete[] ptr;
  ptr = NULL;
}

PtrMatrix::PtrMatrix(int r, int c, int l, double *p, char trans_)
  : mRows(r), mCols(c), leadD(l), ptr(p),
    has_memory(false), trans(trans_) {}

// assume row major storage,
//  which is consistant with blas and lapack layout
double* PtrMatrix::operator()(int r, int c) {
  return &ptr[r+c*leadD];
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

double* PtrMatrix::pointer() const {return ptr;}

int PtrMatrix::LD() const {return leadD;}

int PtrMatrix::rows() const {
  switch (trans) {
  case 'n': return mRows; break;
  case 't': return mCols; break;
  default: assert(false); break;
  }
}

int PtrMatrix::cols() const {
  switch (trans) {
  case 't': return mRows; break;
  case 'n': return mCols; break;
  default: assert(false); break;
  }
}

void PtrMatrix::gemm
(const PtrMatrix& U, const PtrMatrix& V,
 const PtrMatrix& D, const PtrMatrix& res) {
  assert(U.cols() == V.rows());
  double alpha = 1.0, beta = 0.0;
  blas::gemm(U.trans, V.trans, U.rows(), V.cols(), U.cols(),
	     &alpha, U.pointer(), U.LD(),
	     V.pointer(), V.LD(),
	     &beta, res.pointer(), res.LD());
}
  
