#include "tree.hpp"

#include <math.h> // for pow()

void UTree::init(int nProc_, const Matrix& UMat_) {
  this->nProc = nProc_;
  this->UMat  = UMat_;
}

void UTree::init_rhs
(const Vector& b, Context ctx, HighLevelRuntime *runtime) {
  assert(Rhs.rows() == b.rows());
  assert(Rhs.cols() == 1);
  Rhs.init_data(nProc, b, ctx, runtime);
}

Vector UTree::rhs() {
  return Rhs.to_vector();
}

void UTree::partition
(int level, Context ctx, HighLevelRuntime *runtime) {
  

}

LMatrix& UTree::leaf() {
  return level(nlevel-1).uMat;
}

UTree::UDMat& UTree::level(int i) {
  assert( i > 0 );
  assert( i < nlevel );
  return U[i];
}

void VTree::init(int nProc_, const Matrix& VMat_) {
  this->nProc = nProc_;
  this->VMat  = VMat_;  
}

void VTree::partition
(int level, Context ctx, HighLevelRuntime *runtime) {
  // make sure VMat is valid
  assert( VMat.rows() > 0 );
  assert( VMat.cols() > 0 );
  // create region
  V.create(VMat.rows(), VMat.cols(), ctx, runtime);
  // initialize region
  V.init_data(nProc, VMat, ctx, runtime);
  // create partition
  V.partition(nlevel, ctx, runtime);
}

LMatrix& VTree::level(int i) {
  assert( i >= 0 );
  assert( i <  nlevel );
  return V;
}

void KTree::init
(int nProc_, const Matrix& UMat_, const Matrix& VMat_,
 const Vector& DVec_) {
  this->nProc = nProc_;
  this->UMat  = UMat_;
  this->VMat  = VMat_;
  this->DVec  = DVec_;
  assert(nProc == UMat.num_partition());
  assert(nProc == VMat.num_partition());
  assert(nProc == DVec.num_partition());
  assert(UMat.rows() == VMat.rows());
  assert(UMat.cols() == VMat.cols());
  assert(UMat.rows() == DVec.rows());
}

void KTree::partition
(int level, Context ctx, HighLevelRuntime *runtime) {
  // create region
  int nrow = DVec.rows();
  int nblk = pow(2, level-1);
  int ncol = DVec.rows() / nblk;
  assert(nblk >= nProc);
  assert(ncol >  UMat.cols());
  K.create( nrow, ncol, ctx, runtime );
  // initialize region
  K.init_dense_blocks(nProc, nblk, UMat, VMat, DVec, ctx, runtime);
  // partition region
  K.partition(nlevel, ctx, runtime);
}

void KTree::solve
(LMatrix& U, Context ctx, HighLevelRuntime *runtime) {
  K.solve(U, ctx, runtime);
}
