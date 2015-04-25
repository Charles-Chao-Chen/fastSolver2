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
  U.init_data(b, ctx, runtime);
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
  // partition the big region
  // this is the only partition we will use
  // i.e. the same partition for all u and d
  // matrices
  // More precise/complicated partitions can be
  // used, but not necessary in this case.
  this->mLevel = level;
  U.partition(mLevel, ctx, runtime);
  // initialize region
  U.init_data(1, U.cols(), UMat, ctx, runtime);

  // Set column range for all u and d matrics.
  // In particular, we need to set the column begin
  // for u matrices.
  for (int i=0; i<mLevel; i++) {
    int ncol = nRhs + i*rank;
    LMatrix dMat = U;
    dMat.set_column_size(ncol);
    dMat_vec.push_back(dMat);
    LMatrix uMat = U;
    uMat.set_column_size(rank);
    uMat.set_column_begin(ncol);
    uMat_vec.push_back(uMat);
  }  
  assert(uMat_vec.size() == size_t(mLevel));
  assert(dMat_vec.size() == size_t(mLevel));
}

LMatrix& UTree::uMat_level(int i) {
  assert(0<i && i<=mLevel);
  return uMat_vec[i-1];
}

LMatrix& UTree::dMat_level(int i) {
  assert(0<i && i<=mLevel);
  return dMat_vec[i-1];
}

LMatrix& UTree::leaf() {
  return U;
}

Matrix UTree::solution(Context ctx, HighLevelRuntime *runtime) {
  Matrix sln(UMat.rows(), nRhs);
  LogicalRegion lr = U.logical_region();
  RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
  req.add_field(FIELDID_V);
 
  InlineLauncher launcher(req);
  PhysicalRegion region = runtime->map_region(ctx, launcher);
  region.wait_until_valid();
 
  PtrMatrix temp = get_raw_pointer(region, 0, U.rows(), 0, U.cols());
  for (int j=0; j<nRhs; j++)
    for (int i=0; i<sln.rows(); i++)
      sln(i, j) = temp(i, j);
  runtime->unmap_region(ctx, region);
  return sln;
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
  // create partition
  this->mLevel = level;
  V.partition(mLevel, ctx, runtime);
  // initialize region
  V.init_data(VMat, ctx, runtime);
}

LMatrix& VTree::level(int i) {
  assert( 0 < i && i <= mLevel );
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
  // create region
  K.create( nrow, ncol, ctx, runtime );
  // partition region
  this->mLevel = level;
  K.partition(mLevel, ctx, runtime);
  // initialize region
  K.init_dense_blocks(UMat, VMat, DVec, ctx, runtime);
}

void KTree::solve
(LMatrix& U, Context ctx, HighLevelRuntime *runtime) {
  K.solve(U, ctx, runtime);
}
