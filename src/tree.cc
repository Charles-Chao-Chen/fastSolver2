#include "tree.hpp"

#include <math.h> // for pow()

void UTree::init(const Matrix& UMat_) {
  assert(UMat_.rows()>0 && UMat_.cols()>0);
  this->UMat  = UMat_;
  this->rank  = UMat.cols();
  this->nRhs  = 1; // hard code the number of rhs
}

void UTree::init(int level, const Matrix& UMat_,
		 Context ctx, HighLevelRuntime *runtime) {
  assert(UMat_.rows()>0 && UMat_.cols()>0);
  this->mLevel = level;
  this->UMat   = UMat_;
  this->nRhs   = 1; // hard code the number of rhs
  this->rank   = UMat.cols();
  // create the region 
  int cols = nRhs + UMat.cols()*mLevel;
  U.create(UMat.rows(), cols, ctx, runtime);
}

void UTree::init_rhs
(const Vector& b, Context ctx, HighLevelRuntime *runtime) {
  assert(false);
}

void UTree::init_rhs
(const Matrix& b, Context ctx, HighLevelRuntime *runtime,
 bool wait) {
  assert(b.cols()==1);
  U.init_data(b, ctx, runtime, wait);
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
  int cols = nRhs + UMat.cols()*UMat.levels();
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
    int ncol = nRhs + rank*i;
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

void UTree::horizontal_partition
(int task_level, Context ctx, HighLevelRuntime *runtime) {

  // partition data
  U.partition(task_level, ctx, runtime);
  // initialize region
  U.init_data(nRhs, U.cols(), UMat, ctx, runtime);

  // Set column range for all u and d matrics.
  // In particular, we need to set the column begin
  // for u matrices.
  for (int i=0; i<mLevel; i++) {
    int ncol = nRhs + rank*i;
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

LMatrix& UTree::uMat_level_new(int i) {
  assert(0<=i && i<mLevel);
  return uMat_vec[i];
}

LMatrix& UTree::dMat_level_nw(int i) {
  assert(0<=i && i<mLevel);
  return dMat_vec[i];
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

void VTree::init(const Matrix& VMat_) {
  this->VMat  = VMat_;  
}

void VTree::init(int level, const Matrix& VMat_,
		 Context ctx, HighLevelRuntime *runtime) {
  // make sure VMat is valid
  assert( VMat_.rows() > 0 && VMat_.cols() > 0);
  this->mLevel = level;
  this->VMat   = VMat_;  
  // create region
  V.create(VMat.rows(), VMat.cols(), ctx, runtime);
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

void VTree::horizontal_partition
(int task_level, Context ctx, HighLevelRuntime *runtime) {
  // create partition
  V.partition(task_level, ctx, runtime);
  // initialize region
  V.init_data(VMat, ctx, runtime);
}

LMatrix& VTree::leaf() {
  return V;
}

LMatrix& VTree::level(int i) {
  assert( 0 < i && i <= mLevel );
  return V;
}

LMatrix& VTree::level_new(int i) {
  assert( 0 <= i && i < mLevel );
  return V;
}

void KTree::init
(const Matrix& UMat_, const Matrix& VMat_,
 const Vector& DVec_) {
  this->UMat  = UMat_;
  this->VMat  = VMat_;
  this->DVec  = DVec_;
  assert(UMat.rows() == VMat.rows());
  assert(UMat.cols() == VMat.cols());
  assert(UMat.rows() == DVec.rows());
}

void KTree::init
(int level, const Matrix& UMat_, const Matrix& VMat_,  const Vector& DVec_,
 Context ctx, HighLevelRuntime *runtime) {
  this->mLevel = level;
  this->UMat  = UMat_;
  this->VMat  = VMat_;
  this->DVec  = DVec_;
  // check consistancy
  assert(UMat.rows() == VMat.rows());
  assert(UMat.cols() == VMat.cols());
  assert(UMat.rows() == DVec.rows());
  // create region
  int nrow = UMat.rows();
  int nblk = pow(2, UMat.levels());
  int ncol = UMat.rows() / nblk; // leaf size
  K.create( nrow, ncol, ctx, runtime );
}

void KTree::partition
(int level, Context ctx, HighLevelRuntime *runtime) {
  // create region
  int nrow = DVec.rows();
  int nblk = pow(2, UMat.levels());
  int ncol = DVec.rows() / nblk;
  assert(ncol>0);
  K.create( nrow, ncol, ctx, runtime );
  // partition region
  this->mLevel = level;
  K.partition(mLevel, ctx, runtime);
  // initialize region
  K.init_dense_blocks(UMat, VMat, DVec, ctx, runtime, true /*wait*/);
}

void KTree::horizontal_partition
(int task_level, Context ctx, HighLevelRuntime *runtime) {
  // partition region
  K.partition(task_level, ctx, runtime);
  // initialize region
  K.init_dense_blocks(UMat, VMat, DVec, ctx, runtime);
}

void KTree::solve
(LMatrix& U, LMatrix& V, Context ctx, HighLevelRuntime *runtime) {
  K.solve(U, V, ctx, runtime);
}
