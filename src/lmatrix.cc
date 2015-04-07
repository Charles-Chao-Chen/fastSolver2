#include "lmatrix.hpp"
#include "macros.hpp" // for FIELDID_V
#include "solver_tasks.hpp"

LMatrix::LMatrix() {}

int LMatrix::rows() const {return mRows;}

int LMatrix::cols() const {return mCols;}

int LMatrix::num_partition() const {return nPart;}

Domain LMatrix::color_domain() const {return domain;}

LogicalRegion LMatrix::logical_region() const {return region;}

LogicalPartition LMatrix::logical_partition() const {return lpart;}

void LMatrix::init
(const Vector& vec, Context ctx, HighLevelRuntime *runtime, bool wait) {
  assert( this->rows() == vec.rows() );
  assert( this->cols() == 1 );
  assert( this->num_partition() == vec.num_partition() );
  ArgumentMap argMap;
  for (int i = 0; i < nPart; i++) {
    long s = vec.rand_seeds(i);
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
}

void LMatrix::create
(const Matrix& mat, Context ctx, HighLevelRuntime *runtime, bool wait) {
  // create the region
  assert( this->rows() == mat.rows() );
  assert( this->cols() == mat.cols() );
  assert( this->num_partition() == vec.num_partition() );
  ArgumentMap argMap;
  for (int i = 0; i < nPart; i++) {
    long s = mat.rand_seeds(i);
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
}

Vector LMatrix::to_vector() {
  // inline launcher
}

void LMatrix::create_dense_partition
(int nPart, const Matrix& U, const Matrix& V, const Vector& D,
 Context ctx, HighLevelRuntime *runtime, bool wait=WAIT_DEFAULT) {

  // create region and nProc partition
  // populate data
  // create nPart partition
}

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
(const std::string& name, Context ctx, HighLevelRuntime *runtime) {


}
