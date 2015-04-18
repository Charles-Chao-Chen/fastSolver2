#include "ptr_matrix.hpp"
#include "utility.hpp"

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
double PtrMatrix::operator()(int r, int c) const {
  return ptr[r+c*leadD];
}

double& PtrMatrix::operator()(int r, int c) {
  return ptr[r+c*leadD];
}

double* PtrMatrix::pointer() const {return ptr;}

double* PtrMatrix::pointer(int r, int c) {
  return &ptr[r+c*leadD];
}

void PtrMatrix::set_trans(char trans_) {this->trans=trans_;}

void PtrMatrix::clear(double value) {
  for (int j=0; j<mCols; j++)
    for (int i=0; i<mRows; i++)
      (*this)(i, j) = value;
}

void PtrMatrix::scale(double alpha) {
  for (int j=0; j<mCols; j++)
    for (int i=0; i<mRows; i++)
      (*this)(i, j) *= alpha;
}

void PtrMatrix::rand(long seed) {
  struct drand48_data buffer;
  assert( srand48_r( seed, &buffer ) == 0 );
  for (int i=0; i<mRows; i++) {
    for (int j=0; j<mCols; j++) {
      assert( drand48_r(&buffer, this->pointer(i,j) ) == 0 );
    }
  }
}

void PtrMatrix::display(const std::string& name) {
  std::cout << name << ":" << std::endl;
  for(int ri = 0; ri < mRows; ri++) {
    for(int ci = 0; ci < mCols; ci++) {
      std::cout << (*this)(ri, ci) << "\t";
    }
    std::cout << std::endl;
  }
}

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

void PtrMatrix::solve(PtrMatrix& B) {
  int N = this->mRows;
  int NRHS = B.cols();
  int LDA = leadD;
  int LDB = B.LD();
  int IPIV[N];
  int INFO;
  lapack::dgesv_(&N, &NRHS, ptr, &LDA, IPIV,
		 B.pointer(), &LDB, &INFO);
  assert(INFO==0);
  /*
  std::cout << "Permutation:" << std::endl;
  for (int i=0; i<N; i++)
    std::cout << IPIV[i] << "\t";
  std::cout << std::endl;
  */
}

void PtrMatrix::add
  (double alpha, const PtrMatrix& A,
   double beta,  const PtrMatrix& B, PtrMatrix& C) {

  assert(A.rows() == B.rows() && A.rows() == C.rows());
  assert(A.rows() == B.rows() && A.rows() == C.rows());
  assert(A.cols() == B.cols() && A.cols() == C.cols());
  for (int j=0; j<C.cols(); j++)
    for (int i=0; i<C.rows(); i++) {
      C(i, j) = alpha*A(i,j) + beta*B(i,j);
      //printf("(%f, %f, %f)\n", A(i, j), B(i, j), C(i, j));
    }
}

void PtrMatrix::gemm
(const PtrMatrix& U, const PtrMatrix& V, const PtrMatrix& D,
 PtrMatrix& res) {
  assert(U.cols() == V.rows());
  char transa = U.trans;
  char transb = V.trans;
  int  M = U.rows();
  int  N = V.cols();
  int  K = U.cols();
  int  LDA = U.LD();
  int  LDB = V.LD();
  int  LDC = res.LD();
  double alpha = 1.0, beta = 0.0;
  //double alpha = 0.0, beta = 0.0;
  blas::dgemm_(&transa, &transb, &M, &N, &K,
	       &alpha, U.pointer(), &LDA,
	       V.pointer(), &LDB,
	       &beta, res.pointer(), &LDC);

  // add the diagonal
  assert(res.rows() == res.cols());
  for (int i=0; i<res.rows(); i++)
    res(i, i) += D(i, 0);
}
  
void PtrMatrix::gemm
(double alpha, const PtrMatrix& U, const PtrMatrix& V,
 PtrMatrix& W) {
  assert(U.cols() == V.rows());
  assert(U.rows() == W.rows());
  assert(V.cols() == W.cols());
  char transa = U.trans;
  char transb = V.trans;
  int  M = U.rows();
  int  N = V.cols();
  int  K = U.cols();
  int  LDA = U.LD();
  int  LDB = V.LD();
  int  LDC = W.LD();
  double beta = 1.0;
  blas::dgemm_(&transa, &transb, &M, &N, &K,
	       &alpha, U.pointer(), &LDA,
	       V.pointer(), &LDB,
	       &beta, W.pointer(), &LDC);
}
  
