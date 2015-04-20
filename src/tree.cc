#include "tree.hpp"

#include <math.h> // for pow()

void UTree::init(int nProc_, const Matrix& UMat_) {
  assert(UMat_.rows()>0 && UMat_.cols()>0);
  this->nProc = nProc_;
  this->UMat  = UMat_;
  this->rank  = UMat.cols();
  this->nRhs  = 1; // hard code the number of rhs
}

void UTree::init_rhs
(const Vector& b, Context ctx, HighLevelRuntime *runtime) {
  assert(false);
}

void UTree::init_rhs
(const Matrix& b, Context ctx, HighLevelRuntime *runtime) {
  assert(b.cols()==1);
  U.init_data(nProc, 0, 1, b, ctx, runtime);
}

Vector UTree::rhs() {
  assert(false);
  //return Rhs.to_vector();
  return Vector();
}

void UTree::partition
(int level, Context ctx, HighLevelRuntime *runtime) {
  // make sure UMat is valid
  assert( UMat.rows() > 0 );
  assert( UMat.cols() > 0 );
  // create region
  int cols = nRhs + level*UMat.cols();
  U.create(UMat.rows(), cols, ctx, runtime);
  // initialize region
  U.init_data(nProc, 1, cols, UMat, ctx, runtime);
  // create partition
  this->mLevel = level;
  // right hand side partition
  //Rhs = U.partition(/* !LEVEL, NOT NPART*/, 0, nRhs, ctx, runtime);
  for (int i=0; i<mLevel; i++) {
    int ncol = nRhs + i*rank;
    LMatrix dMat = U.partition(mLevel, 0, ncol, ctx, runtime);
    dMat_vec.push_back(dMat);
    LMatrix uMat = U.partition(mLevel, ncol+1, ncol+1+rank, ctx, runtime);
    uMat_vec.push_back(uMat);
  }
  LMatrix leaf = U.partition(mLevel, 0, U.cols(), ctx, runtime);
  dMat_vec.push_back(leaf);
  assert(uMat_vec.size() == size_t(mLevel));
  assert(dMat_vec.size() == size_t(mLevel+1));
}

LMatrix& UTree::uMat_level(int i) {
  assert(0<=i && i<mLevel);
  return uMat_vec[i];
}

LMatrix& UTree::dMat_level(int i) {
  assert(0<=i && i<=mLevel);
  return dMat_vec[i];
}

LMatrix& UTree::leaf() {
  return dMat_vec[mLevel];
}

/*
LMatrix& UTree::leaf() {
  return level(mLevel-1).uMat;
}

UTree::UDMat& UTree::level(int i) {
  assert( i > 0 );
  assert( i < mLevel );
  return Ulevel[i];
}
*/

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
  V.init_data(nProc, 0, VMat.cols(), VMat, ctx, runtime);
  // create partition
  this->mLevel = level;
  V.partition(mLevel, ctx, runtime);
}

LMatrix& VTree::level(int i) {
  assert( i >= 0 );
  assert( i <  mLevel );
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
  int nblk = pow(2, level);
  int ncol = DVec.rows() / nblk;
  assert(nblk >= nProc);
  assert(ncol >= UMat.cols());
  K.create( nrow, ncol, ctx, runtime );
  // initialize region
  K.init_dense_blocks(nProc, nblk, UMat, VMat, DVec, ctx, runtime);
  // partition region
  this->mLevel = level;
  K.partition(mLevel, ctx, runtime);
}

void KTree::solve
(LMatrix& U, Context ctx, HighLevelRuntime *runtime) {
  K.solve(U, ctx, runtime);
}
