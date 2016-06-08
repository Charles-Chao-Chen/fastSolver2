#include "lmatrix.hpp"
#include <math.h> // for pow()

static Realm::Logger log_solver_tasks("solver_tasks");

LMatrix::LMatrix() : nPart(-1) {}

LMatrix::LMatrix
(int rows, int cols, int level,
 Context ctx, HighLevelRuntime *runtime) {
  create(rows, cols, ctx, runtime);
  partition(level, ctx, runtime);
  this->plevel = 1;
}

LMatrix::LMatrix
(int rows, int cols, LogicalRegion r, IndexSpace is, FieldSpace fs)
  : mRows(rows), mCols(cols), ispace(is), fspace(fs), region(r) {}

LMatrix::LMatrix(LogicalRegion r, int rows, int cols) {
  this->region = r;
  this->ispace = region.get_index_space();
  this->mRows  = rows;
  this->mCols  = cols;
}

/*
LMatrix::LMatrix
(int rows, int cols, int part, IndexPartition ip, LogicalRegion lr,
 Context ctx, HighLevelRuntime *runtime)
  : mRows(rows), mCols(cols), colIdx(0),
    nPart(part), rblock(rows/part),
    ipart(ip), region(lr), pregion(lr) {
  this->lpart  = runtime->get_logical_partition(ctx, region, ipart);
  this->colDom = runtime->get_index_partition_color_space(ctx,ipart);
}
*/
LMatrix::~LMatrix() {}

int LMatrix::rows() const {return mRows;}

int LMatrix::cols() const {return mCols;}

int LMatrix::rowBlk() const {return rblock;}

int LMatrix::column_begin() const {return colIdx;}

int LMatrix::num_partition() const {return nPart;}

int LMatrix::partition_level() const {return plevel;}

Domain LMatrix::color_domain() const {
  //return runtime->get_index_partition_color_space(ctx,ipart);
  return colDom;
}

IndexSpace LMatrix::index_space() const {return ispace;}

LogicalRegion LMatrix::logical_region() const {return region;}

IndexPartition LMatrix::index_partition() const {return ipart;}

LogicalPartition LMatrix::logical_partition() const {
  //return runtime->get_logical_partition(ctx, region, ipart);
  return lpart;
}

void LMatrix::set_column_size(int n) {mCols=n;}

void LMatrix::set_column_begin(int begin) {colIdx=begin;}

void LMatrix::set_logical_region(LogicalRegion lr) {region=lr;}

void LMatrix::set_parent_region(LogicalRegion lr) {pregion=lr;}

void LMatrix::set_logical_partition(LogicalPartition lp) {lpart=lp;}

void LMatrix::create
(int rows, int cols, Context ctx, HighLevelRuntime *runtime) {
  assert(rows>0 && cols>0);
  this->mRows = rows;
  this->mCols = cols;
  this->colIdx = 0;
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
  this->region = runtime->create_logical_region(ctx, ispace, fspace);
  //this->parent = region;
  assert(region != LogicalRegion::NO_REGION);
}

void LMatrix::clear
(double value, Context ctx, HighLevelRuntime *runtime, bool wait) {
  
  // assuming partition is done
  assert(nPart > 0);
  ClearMatrixTask::TaskArgs args = {rblock, mCols, value};
  ClearMatrixTask launcher(colDom, TaskArgument(&args, sizeof(args)), ArgumentMap());
  RegionRequirement req(lpart, 0, WRITE_DISCARD, EXCLUSIVE, region);
  req.add_field(FIELDID_V);
  launcher.add_region_requirement(req);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
    
  if(wait) {
    std::cout << "Wait for clearing matrix..." << std::endl;
    fm.wait_all_results();
  }
}

void LMatrix::scale
(double alpha, Context ctx, HighLevelRuntime *runtime, bool wait) {

  // if alpha = 1.0, do nothing
  if ( fabs(alpha - 1.0) < 1e-10) return;
  ScaleMatrixTask::TaskArgs args = {this->rblock * plevel, mCols, alpha};
  TaskArgument tArg(&args, sizeof(args));
  ScaleMatrixTask launcher(colDom, tArg, ArgumentMap(), colDom.get_volume());
  RegionRequirement req(lpart, 0, READ_WRITE, EXCLUSIVE, region);
  //RegionRequirement req(lpart, 0, WRITE_DISCARD, EXCLUSIVE, region);
  req.add_field(FIELDID_V);
  launcher.add_region_requirement(req);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  
  if(wait) {
    log_solver_tasks.print("Wait for scaling matrix...");
    fm.wait_all_results();
    log_solver_tasks.print("Done for scaling matrix...");
  }
}

void LMatrix::init_data
(const Matrix& mat,
 Context ctx, HighLevelRuntime *runtime, bool wait) {
  assert(mCols>=mat.cols());  
  init_data(0, mat.cols(), mat, ctx, runtime, wait);
}

void LMatrix::init_data
(int col0, int col1, const Matrix& mat,
 Context ctx, HighLevelRuntime *runtime, bool wait) {
  assert(col0>=0);
  assert(col1<=mCols);
  assert((col1-col0)>=mat.cols());
  assert((col1-col0)%mat.cols()==0);
  assert(mat.num_partition()%nPart==0);
  this->smallblk = mat.num_partition()/nPart;
  ArgumentMap seeds = MapSeed(mat);
  InitMatrixTask::TaskArgs args = {rblock, mat.cols(), col0, col1};
  TaskArgument tArg(&args, sizeof(args));
  InitMatrixTask launcher(colDom, tArg, seeds, nPart);
  //RegionRequirement req(lpart, 0, WRITE_DISCARD, EXCLUSIVE, region);
  RegionRequirement req(lpart, 0, READ_WRITE, EXCLUSIVE, region);
  req.add_field(FIELDID_V);
  launcher.add_region_requirement(req);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);

  // wait for initialization
  //fm.wait_all_results();
  
  /*
  if(wait) {
    log_solver_tasks.print("Wait for init tree...");
    std::cout<<"Wait for init tree..."<<std::endl;
    fm.wait_all_results();
    log_solver_tasks.print("Done for init tree...");
    std::cout<<"Done for init tree..."<<std::endl;
  }
  */
}

Matrix LMatrix::to_matrix(Context ctx, HighLevelRuntime *runtime) {
  Matrix temp(mRows, mCols);
  RegionRequirement req(region, READ_ONLY, EXCLUSIVE, region);
  req.add_field(FIELDID_V);
 
  InlineLauncher launcher(req);
  PhysicalRegion region = runtime->map_region(ctx, launcher);
  region.wait_until_valid();
 
  PtrMatrix pMat = get_raw_pointer(region, 0, mRows, colIdx, colIdx+mCols);
  for (int j=0; j<mCols; j++)
    for (int i=0; i<mRows; i++)
      temp(i, j) = pMat(i, j);
  runtime->unmap_region(ctx, region);
  return temp;
}
  
Matrix LMatrix::to_matrix
(int col0, int col1, Context ctx, HighLevelRuntime *runtime) {
  assert(col0>=0);
  assert(col1<=mCols);
  Matrix temp(mRows, col1-col0);
  RegionRequirement req(region, READ_ONLY, EXCLUSIVE, region);
  req.add_field(FIELDID_V);
 
  InlineLauncher launcher(req);
  PhysicalRegion region = runtime->map_region(ctx, launcher);
  region.wait_until_valid();
 
  PtrMatrix pMat = get_raw_pointer(region, 0, mRows, col0, col1);
  for (int j=0; j<temp.cols(); j++)
    for (int i=0; i<temp.rows(); i++)
      temp(i, j) = pMat(i, j);
  runtime->unmap_region(ctx, region);
  return temp;
}
  
Matrix LMatrix::to_matrix
(int rlo, int rhi, int clo, int chi,
 Context ctx, HighLevelRuntime *runtime) {
  assert(rlo>=0&&clo>=0);
  assert(rhi<=mRows&&chi<=mCols);
  Matrix temp(rhi-rlo, chi-clo);
  RegionRequirement req(region, READ_ONLY, EXCLUSIVE, region);
  req.add_field(FIELDID_V);
 
  InlineLauncher launcher(req);
  PhysicalRegion region = runtime->map_region(ctx, launcher);
  region.wait_until_valid();
 
  PtrMatrix pMat = get_raw_pointer(region, rlo, rhi, clo, chi);
  for (int j=0; j<temp.cols(); j++)
    for (int i=0; i<temp.rows(); i++)
      temp(i, j) = pMat(i, j);
  runtime->unmap_region(ctx, region);
  return temp;
}
  
// to be removed
/*
void LMatrix::init_data
(int nProc_, const Matrix& mat,
 Context ctx, HighLevelRuntime *runtime, bool wait) {
  assert(mCols==mat.cols());
  init_data(nProc_, 0, mCols, mat, ctx, runtime, wait);
}

void LMatrix::init_data
(int nProc_, int col0, int col1, const Matrix& mat,
 Context ctx, HighLevelRuntime *runtime, bool wait) {
  // assuming the region has been created
  this->nProc = nProc_;  
  ArgumentMap seeds = MapSeed(nProc, mat);
  
  // assume uniform partition
  assert(mRows%nProc == 0);
  IndexPartition ip = UniformRowPartition(nProc, col0, col1, ctx, runtime);
  LogicalPartition lp = runtime->get_logical_partition(ctx, region, ip);
  Domain dom = runtime->get_index_partition_color_space(ctx, ip);

  assert(false);
  InitMatrixTask::TaskArgs args;// = {mRows/nProc, mat.cols(), col0, col1};
  InitMatrixTask launcher(dom, TaskArgument(&args, sizeof(args)), seeds);
  RegionRequirement req(lp, 0, WRITE_DISCARD, EXCLUSIVE, region);
  req.add_field(FIELDID_V);
  launcher.add_region_requirement(req);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
    
  if(wait) {
    std::cout << "Wait for init tree..." << std::endl;
    fm.wait_all_results();
    std::cout << "Done for init tree..." << std::endl;
  }
}
*/
void LMatrix::init_dense_blocks
(const Matrix& U, const Matrix& V, const Vector& D,
 Context ctx, HighLevelRuntime *runtime, bool wait) {
  assert(U.num_partition()%nPart==0);
  assert(V.num_partition()%nPart==0);
  assert(D.num_partition()%nPart==0);
  assert(U.cols()==V.cols());
  assert(U.rows()==V.rows());
  assert(U.rows()==D.rows());
  ArgumentMap seeds = MapSeed(U, V, D);
  int rank = U.cols();
  DenseBlockTask::TaskArgs args = {rblock, rank, D.offset()};
  TaskArgument tArg(&args, sizeof(args));
  DenseBlockTask launcher(colDom, tArg, seeds, this->nPart);
  RegionRequirement req(lpart, 0, WRITE_DISCARD, EXCLUSIVE, region);
  req.add_field(FIELDID_V);
  launcher.add_region_requirement(req);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
    
  if(wait) {
    log_solver_tasks.print("Wait for init dense blocks...");
    fm.wait_all_results();
    log_solver_tasks.print("Done for init dense blocks...");
  }
}
/*
void LMatrix::init_dense_blocks
(int nProc_, int nblk, const Matrix& U, const Matrix& V, const Vector& D,
 Context ctx, HighLevelRuntime *runtime, bool wait) {
  
  this->nProc = nProc_;
  ArgumentMap seeds = MapSeed(nProc, U, V, D);

  // assume uniform partition
  assert(mRows%nProc == 0);
  IndexPartition ip = UniformRowPartition(nProc, 0, mCols, ctx, runtime);
  LogicalPartition lp = runtime->get_logical_partition(ctx, region, ip);
  Domain dom = runtime->get_index_partition_color_space(ctx, ip);

  assert(false);
  DenseBlockTask::TaskArgs args;// = {mRows/nProc, mCols, U.cols(), nblk/nProc};
  DenseBlockTask launcher(dom, TaskArgument(&args, sizeof(args)), seeds);
  RegionRequirement req(lp, 0, WRITE_DISCARD, EXCLUSIVE, region);
  req.add_field(FIELDID_V);
  launcher.add_region_requirement(req);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
    
  if(wait) {
    std::cout << "Wait for init dense blocks..." << std::endl;
    fm.wait_all_results();
    std::cout << "Done for init dense blocks..." << std::endl;
  }
}
*/
ArgumentMap LMatrix::MapSeed(const Matrix& matrix) {
  assert(matrix.num_partition()%nPart==0);
  int blk = matrix.num_partition() / nPart;
  ArgumentMap argMap;
  for (int i = 0; i < nPart; i++) {
    std::vector<long> vec;
    vec.push_back(blk);
    for (int j = 0; j < blk; j++) {
      vec.push_back( matrix.rand_seed(i*blk+j) );
    }
    argMap.set_point(DomainPoint::from_point<1>(Point<1>(i)),
		     TaskArgument(&vec[0],sizeof(long)*(blk+1)));
  }
  return argMap;
}

ArgumentMap LMatrix::MapSeed
(const Matrix& U, const Matrix& V, const Vector& D) {
  assert(U.num_partition()%nPart==0);
  int blk = U.num_partition() / nPart;
  ArgumentMap argMap;
  for (int i = 0; i < nPart; i++) {
    std::vector<long> vec;
    vec.push_back(blk);
    for (int j = 0; j < blk; j++) {
      vec.push_back(U.rand_seed(i*blk+j));
      vec.push_back(V.rand_seed(i*blk+j));
      vec.push_back(D.rand_seed(i*blk+j));
    }
    argMap.set_point(DomainPoint::from_point<1>(Point<1>(i)),
		     TaskArgument(&vec[0],sizeof(long)*(3*blk+1)));
  }
  return argMap;
}
/*
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
*/
IndexPartition LMatrix::UniformRowPartition
(Context ctx, HighLevelRuntime *runtime) {

  Rect<1> bounds(Point<1>(0),Point<1>(nPart-1));
  Domain  domain = Domain::from_rect<1>(bounds);

  int size = rblock;
  DomainColoring coloring;
  for (int i = 0; i < nPart; i++) {
    Point<2> lo = make_point(  i   *size,   0);
    Point<2> hi = make_point( (i+1)*size-1, mCols-1);
    Rect<2> subrect(lo, hi);
    coloring[i] = Domain::from_rect<2>(subrect);
  }
  return runtime->create_index_partition(ctx, ispace, domain, coloring, true);
}

IndexPartition LMatrix::UniformRowPartition
(int num_subregions, int col0, int col1,
 Context ctx, HighLevelRuntime *runtime) {

  Rect<1> bounds(Point<1>(0),Point<1>(num_subregions-1));
  Domain  domain = Domain::from_rect<1>(bounds);

  int size = mRows / num_subregions;
  DomainColoring coloring;
  for (int i = 0; i < num_subregions; i++) {
    Point<2> lo = make_point(  i   *size,   col0);
    Point<2> hi = make_point( (i+1)*size-1, col1-1);
    Rect<2> subrect(lo, hi);
    coloring[i] = Domain::from_rect<2>(subrect);
  }
  return runtime->create_index_partition(ctx, ispace, domain, coloring, true);
}

Vector LMatrix::to_vector() {
  assert(false);
  return Vector();
}

void LMatrix::partition
(int level, Context ctx, HighLevelRuntime *runtime) {
  // if level=1, the number of partition is 2 for V0 and V1
  this->nPart  = pow(2, level);
  this->rblock = mRows/nPart;
  this->ipart  = UniformRowPartition(ctx, runtime);
  //this->ipart  = UniformRowPartition(nPart, 0, mCols, ctx, runtime);
  this->lpart  = runtime->get_logical_partition(ctx, region, ipart);
  this->colDom = runtime->get_index_partition_color_space(ctx, ipart);
}
/*
LMatrix LMatrix::partition
(int level, int col0, int col1, Context ctx, HighLevelRuntime *runtime) {
  // if level=1, the number of partition is 2 for V0 and V1
  int num_subregions = pow(2, level);
  assert(mRows%num_subregions==0);
  IndexPartition ip = UniformRowPartition(num_subregions, col0, col1, ctx, runtime);
  int cols = col1-col0;
  return LMatrix(mRows, cols, num_subregions, ip, region, ctx, runtime); // interface to be modified
}
*/

int LMatrix::small_block_parts() const {return smallblk;}

// solve A x = b for each partition
//  b will be overwritten by x
void LMatrix::solve
(LMatrix& b, LMatrix& V, Context ctx, HighLevelRuntime* runtime, bool wait) {

  // check if the matrix is square
  //assert( this->rblock == this->cols() );

  // check if the dimensions match
  assert( this->rows() == b.rows() &&
	  this->rows() == V.rows() );
  assert( b.cols() > 0 );

  //solve<LeafSolveTask>(b, ctx, runtime, wait);
  // A and b have the same number of partition
  assert( b.num_partition() == nPart );
 
  LogicalPartition APart = this->logical_partition();
  LogicalPartition bPart = b.logical_partition();
  LogicalPartition VPart = V.logical_partition();
  
  LogicalRegion ARegion = this->logical_region();
  LogicalRegion bRegion = b.logical_region();
  LogicalRegion VRegion = V.logical_region();
  
  Domain domain = this->color_domain();
  LeafSolveTask::TaskArgs args = {this->rblock, b.cols(), V.cols(), V.small_block_parts()};
  TaskArgument tArg(&args, sizeof(args));
  LeafSolveTask launcher(domain, tArg, ArgumentMap(), nPart);
  RegionRequirement AReq(APart, 0, READ_ONLY,  EXCLUSIVE, ARegion);
  RegionRequirement bReq(bPart, 0, READ_WRITE, EXCLUSIVE, bRegion);
  RegionRequirement VReq(VPart, 0, READ_ONLY,  EXCLUSIVE, VRegion);
  AReq.add_field(FIELDID_V);
  bReq.add_field(FIELDID_V);
  VReq.add_field(FIELDID_V);
  launcher.add_region_requirement(AReq);
  launcher.add_region_requirement(bReq);
  launcher.add_region_requirement(VReq);
    
  FutureMap fm = runtime->execute_index_space(ctx, launcher);

  if(wait) {
    log_solver_tasks.print("Wait for leaf solve...");
    fm.wait_all_results();
    log_solver_tasks.print("Done for leaf solve...");
  }
}

void LMatrix::two_level_partition
(Context ctx, HighLevelRuntime *runtime) {
  
  // partition each subregion into two pieces
  // for V0Tu0 and V1Tu1
  for (int i=0; i<nPart; i++) {
    LogicalRegion lr = runtime->get_logical_subregion_by_color(ctx, lpart, i);

    Rect<1> bounds(Point<1>(0),Point<1>(1));
    Domain  domain = Domain::from_rect<1>(bounds);
    int size = rblock/2;
    DomainColoring coloring;
    for (int j = 0; j < 2; j++) {
      Point<2> lo = make_point( i*rblock+j*size,   0);
      Point<2> hi = make_point( i*rblock+(j+1)*size-1, mCols-1);
      Rect<2> subrect(lo, hi);
      coloring[j] = Domain::from_rect<2>(subrect);
    }
    IndexSpace is = lr.get_index_space();
    IndexPartition ip = runtime->create_index_partition(ctx, is, domain, coloring, true,0);
    LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
    (void)lp;
  }
  
  // update partition parameters
  this->plevel  = 2;
  this->rblock /= 2;
  this->nPart  *= 2;
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

  //--------------------------------------------------------------
  // node solve is always launched at the first level of partition
  //--------------------------------------------------------------
  
  //solve<NodeSolveTask>(b, ctx, runtime, wait);

  // A and b have the same number of partition
  // this should refer to the second level
  assert( this->num_partition() == b.num_partition() );

  // rowBlk is the size of the Shur complement,
  // so rowBlk = 2*rowBlk() when plevel=2,
  // or rowBlk = rowBlk() when plevel=1.
  int rowBlk = this->rowBlk()*plevel;
  //std::cout<<"rowBlk:"<<rowBlk<<", mCols:"<<mCols<<std::endl;
  assert( rowBlk/2 == mCols );
  
  // first level stuff
  LogicalPartition APart = this->logical_partition();
  LogicalPartition bPart = b.logical_partition();

  LogicalRegion ARegion = this->logical_region();
  LogicalRegion bRegion = b.logical_region();

  Domain domain = this->color_domain();
  NodeSolveTask::TaskArgs args = {rowBlk, mCols, b.cols()};
  NodeSolveTask launcher(domain, TaskArgument(&args, sizeof(args)),
			 ArgumentMap(), domain.get_volume());
  //RegionRequirement AReq(APart, 0, READ_ONLY,  EXCLUSIVE, ARegion);
  // bug here: have to use stronger previlige
  RegionRequirement AReq(APart, 0, READ_WRITE,  EXCLUSIVE, ARegion);
  RegionRequirement bReq(bPart, 0, READ_WRITE, EXCLUSIVE, bRegion);
  AReq.add_field(FIELDID_V);
  bReq.add_field(FIELDID_V);
  launcher.add_region_requirement(AReq);
  launcher.add_region_requirement(bReq);
  
  FutureMap fm = runtime->execute_index_space(ctx, launcher);

  if(wait) {
    log_solver_tasks.print("Wait for node solve...");
    fm.wait_all_results();
    log_solver_tasks.print("Done for node solve...");
  }
}

void LMatrix::node_solve
(LMatrix& VTu0, LMatrix &VTu1, LMatrix& VTd0, LMatrix &VTd1,
 PhaseBarrier pb_wait, PhaseBarrier pb_ready,
 Context ctx, HighLevelRuntime* runtime, bool wait) {

  LogicalRegion VTu0_rg = VTu0.logical_region();
  LogicalRegion VTu1_rg = VTu1.logical_region();
  LogicalRegion VTd0_rg = VTd0.logical_region();
  LogicalRegion VTd1_rg = VTd1.logical_region();
#if 0
  printf("VTu0 (%x,%x,%x)\n",
	 VTu0_rg.get_index_space().get_id(), 
	 VTu0_rg.get_field_space().get_id(),
	 VTu0_rg.get_tree_id());
  printf("VTu1 (%x,%x,%x)\n",
	 VTu1_rg.get_index_space().get_id(), 
	 VTu1_rg.get_field_space().get_id(),
	 VTu1_rg.get_tree_id());
  printf("VTd0 (%x,%x,%x)\n",
	 VTd0_rg.get_index_space().get_id(), 
	 VTd0_rg.get_field_space().get_id(),
	 VTd0_rg.get_tree_id());
  printf("VTd1 (%x,%x,%x)\n",
	 VTd1_rg.get_index_space().get_id(), 
	 VTd1_rg.get_field_space().get_id(),
	 VTd1_rg.get_tree_id());
#endif
  assert(VTu0.rows() == VTu0.cols());
  assert(VTu1.rows() == VTu1.cols());
  assert(VTu0.rows() == VTu1.rows());
  assert(VTd0.rows() == VTd1.rows());
  assert(VTd0.cols() == VTd1.cols());
  assert(VTu0.rows() == VTd0.rows());
  int rank = VTd0.rows();
  int nRhs = VTd0.cols();
  NodeSolveRegionTask::TaskArgs args = {rank, nRhs};
  NodeSolveRegionTask launcher(TaskArgument(&args, sizeof(args)));
  //RegionRequirement AReq(ARegion, 0, READ_ONLY,  EXCLUSIVE, ARegion);
  RegionRequirement VTu0_rq(VTu0_rg, READ_ONLY, EXCLUSIVE, VTu0_rg);
  RegionRequirement VTu1_rq(VTu1_rg, READ_ONLY, EXCLUSIVE, VTu1_rg);
  RegionRequirement VTd0_rq(VTd0_rg, READ_WRITE, EXCLUSIVE, VTd0_rg);
  RegionRequirement VTd1_rq(VTd1_rg, READ_WRITE, EXCLUSIVE, VTd1_rg);
  VTu0_rq.add_field(FIELDID_V);
  VTu1_rq.add_field(FIELDID_V);
  VTd0_rq.add_field(FIELDID_V);
  VTd1_rq.add_field(FIELDID_V);
  launcher.add_region_requirement(VTu0_rq);
  launcher.add_region_requirement(VTu1_rq);
  launcher.add_region_requirement(VTd0_rq);
  launcher.add_region_requirement(VTd1_rq);
  launcher.add_wait_barrier(pb_wait);
  launcher.add_arrival_barrier(pb_ready);
  runtime->execute_task(ctx, launcher);
}

/*
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
*/
void LMatrix::add
(double alpha, const LMatrix& A,
 double beta, const LMatrix& B, LMatrix& C,
 Context ctx, HighLevelRuntime *runtime, bool wait) {

  // A, B and C have the same size
  assert( A.rows() == B.rows() && A.rows() == C.rows() );
  assert( A.cols() == B.cols() && A.cols() == C.cols() );
  assert( A.num_partition() == B.num_partition() &&
	  A.num_partition() == C.num_partition() );

  LogicalPartition APart = A.logical_partition();
  LogicalPartition BPart = B.logical_partition();
  LogicalPartition CPart = C.logical_partition();

  LogicalRegion AReg = A.logical_region();
  LogicalRegion BReg = B.logical_region();
  LogicalRegion CReg = C.logical_region();

  int rblock = A.rowBlk();
  int cols   = A.cols();
  AddMatrixTask::TaskArgs args = {alpha, beta, rblock, cols};
  TaskArgument tArgs(&args, sizeof(args));
  Domain domain = A.color_domain();
  AddMatrixTask launcher(domain, tArgs, ArgumentMap());  
  RegionRequirement AReq(APart, 0, READ_ONLY, EXCLUSIVE, AReg);
  RegionRequirement BReq(BPart, 0, READ_ONLY, EXCLUSIVE, BReg);
  RegionRequirement CReq(CPart, 0, WRITE_DISCARD, EXCLUSIVE, CReg);
  AReq.add_field(FIELDID_V);
  BReq.add_field(FIELDID_V);
  CReq.add_field(FIELDID_V);
  launcher.add_region_requirement(AReq);
  launcher.add_region_requirement(BReq);
  launcher.add_region_requirement(CReq);
  
  FutureMap fm = runtime->execute_index_space(ctx, launcher);

  if(wait) {
    std::cout << "Wait for adding matrix..." << std::endl;
    fm.wait_all_results();
    std::cout << "Done for adding matrix..." << std::endl;
  }  
}

void LMatrix::gemmRed // static method
(char transa, char transb, double alpha,
 const LMatrix& A, const LMatrix& B,
 double beta, LMatrix& C,
 Context ctx, HighLevelRuntime *runtime, bool wait) {

  assert( fabs(beta - 0.0) < 1e-10);
  C.scale(beta, ctx, runtime);
  
  // A and B have the same number of partition
  assert( A.num_partition() == B.num_partition() );
  assert( A.num_partition() %  C.num_partition() == 0 );

  LogicalPartition APart = A.logical_partition();
  LogicalPartition BPart = B.logical_partition();
  LogicalPartition CPart = C.logical_partition();

  LogicalRegion AReg = A.logical_region();
  LogicalRegion BReg = B.logical_region();
  LogicalRegion CReg = C.logical_region();

  int colorSize = A.num_partition() / C.num_partition();
  GemmRedTask::TaskArgs args={colorSize, C.partition_level(),
			      alpha, transa, transb,
			      A.rowBlk(), B.rowBlk(), C.rowBlk(),
			      A.cols(), B.cols(), C.cols(),
			      A.column_begin(), B.column_begin(), C.column_begin()};
  TaskArgument tArgs(&args, sizeof(args));
  Domain domain = A.color_domain();
  GemmRedTask launcher(domain, tArgs, ArgumentMap(), A.nPart);
  
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
    log_solver_tasks.print("Wait for gemm reduce...");
    fm.wait_all_results();
    log_solver_tasks.print("Done for gemm reduce...");
  }  
}

void LMatrix::gemm // static method
(char transa, char transb,
 double alpha, const LMatrix& A, const LMatrix& B,
 double beta, LMatrix& C,
 Context ctx, HighLevelRuntime *runtime, bool wait) {
  // skip scaling C matrix
  assert( fabs(beta - 0.0) < 1e-10);
  GemmTask::TaskArgs args = {transa, transb, alpha, beta,
			     A.rows(), B.rows(), C.rows(),
			     A.column_begin(), B.column_begin(),
			     A.cols(), B.cols(), C.cols()};
  GemmTask launcher(TaskArgument(&args, sizeof(args)));
  launcher.add_region_requirement
    (RegionRequirement(A.logical_region(),READ_ONLY,EXCLUSIVE,A.logical_region())
     .add_field(FIELDID_V));
  launcher.add_region_requirement
    (RegionRequirement(B.logical_region(),READ_ONLY,EXCLUSIVE,B.logical_region())
     .add_field(FIELDID_V));
  launcher.add_region_requirement
    (RegionRequirement(C.logical_region(),READ_WRITE,EXCLUSIVE,C.logical_region())
     .add_field(FIELDID_V));
  runtime->execute_task(ctx, launcher);
}

void LMatrix::gemm_inplace // static method
(char transa, char transb, double alpha,
 const LMatrix& A, const LMatrix& B,
 double beta, LMatrix& C,
 Context ctx, HighLevelRuntime *runtime, bool wait) {
  // skip scaling C matrix
  assert( fabs(beta - 1.0) < 1e-10);
  GemmInplaceTask::TaskArgs args = {transa, transb, alpha, beta,
				    A.rows(), B.rows(), C.rows(),
				    A.cols(), B.cols(), C.cols(),
				    A.column_begin()};
  GemmInplaceTask launcher(TaskArgument(&args, sizeof(args)));
  launcher.add_region_requirement
    (RegionRequirement(A.logical_region(),READ_WRITE,EXCLUSIVE,A.logical_region())
     .add_field(FIELDID_V));
  launcher.add_region_requirement
    (RegionRequirement(B.logical_region(),READ_ONLY,EXCLUSIVE,B.logical_region())
     .add_field(FIELDID_V));
  //launcher.add_region_requirement
  //  (RegionRequirement(C.logical_region(),READ_WRITE,EXCLUSIVE,C.logical_region())
  //    .add_field(FIELDID_V));
  runtime->execute_task(ctx, launcher);
}

// compute A * B = C; broadcast B
// this is hard coded in that A and C are the same region
// so is GemmBroTask.
void LMatrix::gemmBro // static method
(char transa, char transb, double alpha,
 const LMatrix& A, const LMatrix& B,
 double beta, LMatrix& C,
 Context ctx, HighLevelRuntime *runtime, bool wait) {

  assert( fabs(beta - 1.0) < 1e-10);
  //C.scale(beta, ctx, runtime);
  
  // A and C have the same number of partition
  assert( A.num_partition() == C.num_partition() );
  assert( A.num_partition() %  B.num_partition() == 0 );
  
  LogicalPartition AP = A.logical_partition();
  LogicalPartition BP = B.logical_partition();
  LogicalPartition CP = C.logical_partition();

  LogicalRegion AReg = A.logical_region();
  LogicalRegion BReg = B.logical_region();
  LogicalRegion CReg = C.logical_region();
  assert(AReg==CReg);
  
  int colorSize = A.nPart / B.nPart;
  GemmBroTask::TaskArgs args = {colorSize, B.partition_level(),
				alpha, transa, transb,
				A.rowBlk(), B.rowBlk(), C.rowBlk(),
				A.cols(), B.cols(), C.cols(),
				A.column_begin()};
  TaskArgument tArgs(&args, sizeof(args));
  Domain domain = A.color_domain();
  GemmBroTask launcher(domain, tArgs, ArgumentMap(), A.nPart);
  
  //RegionRequirement AReq(AP, 0,           READ_ONLY,  EXCLUSIVE, AReg);
  RegionRequirement AReq(AP, 0,           READ_WRITE,  EXCLUSIVE, AReg);
  RegionRequirement BReq(BP, CONTRACTION, READ_ONLY,  EXCLUSIVE, BReg);
  //RegionRequirement BReq(BP, CONTRACTION, READ_WRITE,  EXCLUSIVE, BReg);
  RegionRequirement CReq(CP, 0,           READ_WRITE, EXCLUSIVE, CReg);
  AReq.add_field(FIELDID_V);
  BReq.add_field(FIELDID_V);
  CReq.add_field(FIELDID_V);
  launcher.add_region_requirement(AReq);
  launcher.add_region_requirement(BReq);
  //launcher.add_region_requirement(CReq);
  
  FutureMap fm = runtime->execute_index_space(ctx, launcher);

  if(wait) {
    log_solver_tasks.print("Wait for gemm broadcast...");
    fm.wait_all_results();
    log_solver_tasks.print("Done for gemm broadcast...");
  }  
}
  
void LMatrix::display
(const std::string& name,
 Context ctx, HighLevelRuntime *runtime, bool wait) {

  Domain dom = runtime->get_index_space_domain(ctx, region.get_index_space());
  assert(colIdx+mCols<=dom.get_rect<2>().dim_size(1));
  DisplayMatrixTask::TaskArgs args(name, mRows, mCols, colIdx);
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

void LMatrix::clear(Context ctx, HighLevelRuntime* runtime) {
  runtime->destroy_logical_region(ctx, region);
  runtime->destroy_field_space(ctx, fspace);
  runtime->destroy_index_space(ctx, ispace);
}
