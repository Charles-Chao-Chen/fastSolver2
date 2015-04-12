#include "lmatrix.hpp"

LMatrix::LMatrix() {}

LMatrix::~LMatrix() {}

int LMatrix::rows() const {return mRows;}

int LMatrix::cols() const {return mCols;}

int LMatrix::num_partition() const {return nPart;}

Domain LMatrix::color_domain() const {return colDom;}

LogicalRegion LMatrix::logical_region() const {return region;}

LogicalPartition LMatrix::logical_partition() const {return lpart;}

void LMatrix::create
(int rows, int cols, Context ctx, HighLevelRuntime *runtime, bool wait) {
  assert(rows>0 && cols>0);
  this->mRows = rows;
  this->mCols = cols;
  Point<2> lo = make_point(0, 0);
  Point<2> hi = make_point(mRows-1, mCols-1);
  Rect<2> rect(lo, hi);
  this->fspace = runtime->create_field_space(ctx);
  this->ispace = runtime->
    create_index_space(ctx, Domain::from_rect<2>(rect));
  {
    FieldAllocator allocator = runtime->
      create_field_allocator(ctx, fspace);
    allocator.allocate_field(sizeof(double), FIELDID_V);
  }
  region = runtime->create_logical_region(ctx, ispace, fspace);
  assert(region != LogicalRegion::NO_REGION);
}

void LMatrix::init_data
(int nPart, const Vector& vec,
 Context ctx, HighLevelRuntime *runtime, bool wait) {
  /*
  assert( this->rows() == vec.rows() );
  assert( this->cols() == 1 );
  assert( this->num_partition() == vec.num_partition() );
  ArgumentMap argMap;
  for (int i = 0; i < nPart; i++) {
    long s = vec.rand_seed(i);
    argMap.set_point(DomainPoint::from_point<1>(Point<1>(i)),
		     TaskArgument(&s,sizeof(s)));
  }  
  InitMatrixTask launcher(domain, TaskArgument(), argMap);
  RegionRequirement req(lpart, 0, WRITE_DISCARD, EXCLUSIVE, region);
  req.add_field(FIELDID_V);
  launcher.add_region_requirement(req);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
    
  if(wait) {
    std::cout << "Wait for init..." << std::endl;
    fm.wait_all_results();
  }
  */
}

void LMatrix::init_data
(int nProc_, const Matrix& mat,
 Context ctx, HighLevelRuntime *runtime, bool wait) {
  // assuming the region has been created
  assert( this->rows() == mat.rows() );
  assert( this->cols() == mat.cols() );
  this->nProc = nProc_;
  ArgumentMap seeds = MapSeed(nProc, mat);

  // assume uniform partition
  assert(mRows%nProc == 0);
  IndexPartition ip = UniformRowPartition(nProc, ctx, runtime);
  LogicalPartition lp = runtime->get_logical_partition(ctx, region, ip);
  Domain dom = runtime->get_index_partition_color_space(ctx, ip);

  InitMatrixTask::TaskArgs args = {mRows / nProc, mCols};
  InitMatrixTask launcher(dom, TaskArgument(&args, sizeof(args)), seeds);
  RegionRequirement req(lp, 0, WRITE_DISCARD, EXCLUSIVE, region);
  req.add_field(FIELDID_V);
  launcher.add_region_requirement(req);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
    
  if(wait) {
    std::cout << "Wait for init..." << std::endl;
    fm.wait_all_results();
  }
}

void LMatrix::init_dense_blocks
(int nProc_, int nblk, const Matrix& U, const Matrix& V, const Vector& D,
 Context ctx, HighLevelRuntime *runtime, bool wait) {
  
  this->nProc = nProc_;
  ArgumentMap seeds = MapSeed(nProc, U, V, D);

  // assume uniform partition
  assert(mRows%nProc == 0);
  IndexPartition ip = UniformRowPartition(nProc, ctx, runtime);
  LogicalPartition lp = runtime->get_logical_partition(ctx, region, ip);
  Domain dom = runtime->get_index_partition_color_space(ctx, ip);

  DenseBlockTask::TaskArgs args = {mRows/nProc, mCols, U.cols(), nblk/nProc};
  DenseBlockTask launcher(dom, TaskArgument(&args, sizeof(args)), seeds);
  RegionRequirement req(lp, 0, WRITE_DISCARD, EXCLUSIVE, region);
  req.add_field(FIELDID_V);
  launcher.add_region_requirement(req);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
    
  if(wait) {
    std::cout << "Wait for init..." << std::endl;
    fm.wait_all_results();
  }
}

ArgumentMap LMatrix::MapSeed(int nPart, const Matrix& matrix) {
  assert(nPart == matrix.num_partition());
  ArgumentMap argMap;
  for (int i = 0; i < nPart; i++) {
    long s = matrix.rand_seed(i);
    argMap.set_point(DomainPoint::from_point<1>(Point<1>(i)),
		     TaskArgument(&s,sizeof(s)));
  }
  return argMap;
}

ArgumentMap LMatrix::MapSeed(int nPart, const Matrix& U, const Matrix& V, const Vector& D) {
  ArgumentMap argMap;
  for (int i = 0; i < nPart; i++) {
    ThreeSeeds  threeSeeds = {U.rand_seed(i), V.rand_seed(i), D.rand_seed(i)};
    argMap.set_point(DomainPoint::from_point<1>(Point<1>(i)),
		     TaskArgument(&threeSeeds,sizeof(threeSeeds)));
  }
  return argMap;
}

IndexPartition LMatrix::UniformRowPartition
(int num_subregions, Context ctx, HighLevelRuntime *runtime) {

  Rect<1> bounds(Point<1>(0),Point<1>(num_subregions-1));
  Domain  domain = Domain::from_rect<1>(bounds);

  int size = mRows / num_subregions;
  DomainColoring coloring;
  for (int i = 0; i < num_subregions; i++) {
    Point<2> lo = make_point(  i   *size,   0);
    Point<2> hi = make_point( (i+1)*size-1, mCols-1);
    Rect<2> subrect(lo, hi);
    coloring[i] = Domain::from_rect<2>(subrect);
  }
  return runtime->create_index_partition(ctx, ispace, domain, coloring, true);
}

Vector LMatrix::to_vector() {
  // inline launcher
  return Vector();
}

/*
void LMatrix::create_dense_partition
(int nPart, const Matrix& U, const Matrix& V, const Vector& D,
 Context ctx, HighLevelRuntime *runtime, bool wait) {

  // create region and nProc partition
  // populate data
  // create nPart partition
}
*/

void LMatrix::partition
(int level, Context ctx, HighLevelRuntime *runtime) {
  
}

void LMatrix::coarse_partition() {

}

void LMatrix::fine_partition() {

}

// solve A x = b for each partition
//  b will be overwritten by x
void LMatrix::solve
(LMatrix& b, Context ctx, HighLevelRuntime* runtime, bool wait) {

  // check if the matrix is square
  assert( this->rows() == this->cols() );

  // check if the dimensions match
  assert( this->rows() == b.rows() );
  assert( b.cols() > 0 );

  solve<LeafSolveTask>(b, ctx, runtime, wait);
}

// solve the following system for every partition
// --             --  --    --     --      --
// |  I     V1'*u1 |  | eta0 |     | V1'*d1 |
// |               |  |      |  =  |        |
// | V0'*u0   I    |  | eta1 |     | V0'*d0 |
// --             --  --    --     --      --
//  VTd will be overwritten by eta
// Note VTd needs to be reordered,
//  as shown in the above picture.
void LMatrix::node_solve
(LMatrix& b, Context ctx, HighLevelRuntime* runtime, bool wait) {

  // pair up neighbor cells
  this->coarse_partition();
  b.coarse_partition();
  solve<NodeSolveTask>(b, ctx, runtime, wait);
  // recover the original partition
  b.fine_partition();
}

template <typename SolveTask>
void LMatrix::solve
(LMatrix& b, Context ctx, HighLevelRuntime *runtime, bool wait) {

  // A and b have the same number of partition
  assert( this->num_partition() == b.num_partition() );

  LogicalPartition APart = this->logical_partition();
  LogicalPartition bPart = b.logical_partition();

  LogicalRegion ARegion = this->logical_region();
  LogicalRegion bRegion = b.logical_region();

  Domain domain = this->color_domain();
  SolveTask launcher(domain, TaskArgument(), ArgumentMap());
  RegionRequirement AReq(APart, 0, READ_ONLY,  EXCLUSIVE, ARegion);
  RegionRequirement bReq(bPart, 0, READ_WRITE, EXCLUSIVE, bRegion);
  AReq.add_field(FIELDID_V);
  bReq.add_field(FIELDID_V);
  launcher.add_region_requirement(AReq);
  launcher.add_region_requirement(bReq);
  
  FutureMap fm = runtime->execute_index_space(ctx, launcher);

  if(wait) {
    std::cout << "Wait for solve..." << std::endl;
    fm.wait_all_results();
  }
}

// compute A.transpose() * B and reduce to C
void LMatrix::gemmRed // static method
(double alpha, const LMatrix& A, const LMatrix& B,
 double beta,  const LMatrix& C,
 Context ctx, HighLevelRuntime *runtime, bool wait) {

  // A and B have the same number of partition
  assert( A.num_partition() == B.num_partition() );

  LogicalPartition APart = A.logical_partition();
  LogicalPartition BPart = B.logical_partition();
  LogicalPartition CPart = C.logical_partition();

  LogicalRegion AReg = A.logical_region();
  LogicalRegion BReg = B.logical_region();
  LogicalRegion CReg = C.logical_region();

  assert( A.nPart % B.nPart == 0 );
  Domain domain = A.color_domain();
  int colorSize = A.nPart / B.nPart;
  GemmRedTask::TaskArgs args = {colorSize, alpha, beta};
  TaskArgument tArgs(&args, sizeof(args));
  GemmRedTask launcher(domain, tArgs, ArgumentMap());
  
  RegionRequirement AReq(APart, 0,           READ_ONLY, EXCLUSIVE, AReg);
  RegionRequirement BReq(BPart, 0,           READ_ONLY, EXCLUSIVE, BReg);
  RegionRequirement CReq(CPart, CONTRACTION, REDOP_ADD, EXCLUSIVE, CReg);
  AReq.add_field(FIELDID_V);
  BReq.add_field(FIELDID_V);
  CReq.add_field(FIELDID_V);
  launcher.add_region_requirement(AReq);
  launcher.add_region_requirement(BReq);
  launcher.add_region_requirement(CReq);
  
  FutureMap fm = runtime->execute_index_space(ctx, launcher);

  if(wait) {
    std::cout << "Wait for solve..." << std::endl;
    fm.wait_all_results();
  }  
}
  
// compute A * B; broadcast B
void LMatrix::gemmBro // static method
(double alpha, const LMatrix& A, const LMatrix& B,
 double beta,  const LMatrix& C,
 Context ctx, HighLevelRuntime *runtime, bool wait) {

  // A and C have the same number of partition
  assert( A.num_partition() == C.num_partition() );

  LogicalPartition AP = A.logical_partition();
  LogicalPartition BP = B.logical_partition();
  LogicalPartition CP = C.logical_partition();

  LogicalRegion AReg = A.logical_region();
  LogicalRegion BReg = B.logical_region();
  LogicalRegion CReg = C.logical_region();

  assert( A.nPart % B.nPart == 0 );
  Domain domain = A.color_domain();
  int colorSize = A.nPart / B.nPart;
  GemmBroTask::TaskArgs args = {colorSize, alpha, beta};
  TaskArgument tArgs(&args, sizeof(args));
  GemmRedTask launcher(domain, tArgs, ArgumentMap());
  
  RegionRequirement AReq(AP, 0,           READ_ONLY,  EXCLUSIVE, AReg);
  RegionRequirement BReq(BP, CONTRACTION, READ_ONLY,  EXCLUSIVE, BReg);
  RegionRequirement CReq(CP, 0,           READ_WRITE, EXCLUSIVE, CReg);
  AReq.add_field(FIELDID_V);
  BReq.add_field(FIELDID_V);
  CReq.add_field(FIELDID_V);
  launcher.add_region_requirement(AReq);
  launcher.add_region_requirement(BReq);
  launcher.add_region_requirement(CReq);
  
  FutureMap fm = runtime->execute_index_space(ctx, launcher);

  if(wait) {
    std::cout << "Wait for solve..." << std::endl;
    fm.wait_all_results();
  }  
}
  
void LMatrix::display
(const std::string& name,
 Context ctx, HighLevelRuntime *runtime, bool wait) {
  DisplayMatrixTask::TaskArgs args(name, mRows, mCols);
  DisplayMatrixTask launcher(TaskArgument(&args, sizeof(args)));
  RegionRequirement req(region, READ_ONLY, EXCLUSIVE, region);
  req.add_field(FIELDID_V);
  launcher.add_region_requirement(req);
  Future f = runtime->execute_task(ctx, launcher);

  if (wait) {
    std::cout << "Waiting for displaying matrix ..." << std::endl;
    f.get_void_result();
  }
}
