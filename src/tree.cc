#include "tree.hpp"

void UTree::init(int nProc_, const Matrix& UMat_) {
  this->nProc = nProc_;
  this->UMat  = UMat_;
}

void UTree::init_rhs(const Vector& Rhs) {
  level(0).dMat.init(Rhs);
}

Vector UTree::rhs() {
  return level(0).dMat.to_vector();
}

void UTree::partition
(int level, Context ctx, HighLevelRuntime *runtime) {
  // no implemented
  assert( false );

}

LMatrix UTree::leaf() {
  return level(nlevel-1).uMat;
}

UDMat UTree::level(int i) const {
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
  // make sure VMat is initialized
  assert( VMat.rows() > 0 );
  assert( VMat.cols() > 0 );
  V.create(VMat, ctx, runtime);
  V.partition(nlevel, ctx, runtime);
}

LMatrix VTree::level(int i) const {
  assert( i > 0 );
  assert( i < nlevel );
  return V;
}

void KTree::init
(int nProc_, const Matrix& UMat_, const Matrix& VMat_,
 const Vector& DVec_) {
  this->nProc = nProc_;
  this->UMat  = UMat_;
  this->VMat  = VMat_;
  this->KMat  = KVec_;
}

void KTree::partition
(int level, Context ctx, HighLevelRuntime *runtime) {
  K.create_dense_blocks(nlevel, UMat, VMat, Dvec, ctx, runtime);
}

void KTree::solve
(const LMatrix& U, Context ctx, HighLevelRuntime *runtime) {
  K.solve(U);
}
