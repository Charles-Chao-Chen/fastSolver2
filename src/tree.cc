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
    //LMatrix uMat = U.partition(mLevel, ncol, ncol+rank, ctx, runtime);
    //uMat_vec.push_back(uMat);
  }
  LMatrix leaf = U.partition(mLevel, 0, U.cols(), ctx, runtime);
  dMat_vec.push_back(leaf);


  
  LogicalPartition lp;
  {
    // partition U along column
    Rect<1> bounds(Point<1>(0),Point<1>(mLevel-1));
    Domain  domain = Domain::from_rect<1>(bounds);
    int size = this->rank;
    DomainColoring coloring;
    for (int i = 0; i < mLevel; i++) {
      Point<2> lo = make_point( 0,          nRhs+size*i);
      Point<2> hi = make_point( U.rows()-1, nRhs+size*(i+1)-1);
      Rect<2> subrect(lo, hi);
      coloring[i] = Domain::from_rect<2>(subrect);
    }
    IndexPartition ip = runtime->create_index_partition(ctx,
							U.index_space(), domain, coloring, true);

    lp = runtime->get_logical_partition(ctx, U.logical_region(), ip);
  }

  for (int i=0; i<mLevel; i++) {
    LogicalRegion lr = runtime->get_logical_subregion_by_color(ctx, lp, i);

    IndexSpace is = lr.get_index_space();

    
    // partition for uMat
    int num_subregions = pow(2, mLevel);
    
    Rect<1> bounds(Point<1>(0),Point<1>(num_subregions-1));
    Domain  domain = Domain::from_rect<1>(bounds);

    int size = U.rows() / num_subregions;
    DomainColoring coloring;
    for (int i = 0; i < num_subregions; i++) {
      Point<2> lo = make_point(  i   *size,   0);
      Point<2> hi = make_point( (i+1)*size-1, rank);
      Rect<2> subrect(lo, hi);
      coloring[i] = Domain::from_rect<2>(subrect);
    }
    IndexPartition ip = runtime->create_index_partition(ctx, is, domain, coloring, true);
    //LogicalPartition lp = runtime->get_logical_partition(ctx, U.logical_region(), ip);
    //int cols = col1-col0;
    LMatrix uMat(U.rows(), rank, num_subregions, ip, lr, ctx, runtime); // interface to be modified
    //uMat.set_parent_region(U.logical_region());
    uMat_vec.push_back(uMat);


  

    //uMat_vec[i].set_logical_region(lr);
    //uMat_vec[i].set_logical_partition(lpart);
  }

  
  assert(uMat_vec.size() == size_t(mLevel));
  assert(dMat_vec.size() == size_t(mLevel+1));
}

LMatrix& UTree::uMat_level(int i) {
  assert(0<i && i<=mLevel);
  return uMat_vec[i-1];
}

LMatrix& UTree::dMat_level(int i) {
  assert(0<i && i<=mLevel+1);
  return dMat_vec[i-1];
}

LMatrix& UTree::leaf() {
  return dMat_vec[mLevel];
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
  // initialize region
  V.init_data(nProc, 0, VMat.cols(), VMat, ctx, runtime);
  // create partition
  this->mLevel = level;
  V.partition(mLevel, ctx, runtime);
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
