#include "lmatrix.hpp"
#include "macros.hpp"

LMatrix::LMatrix() {}

int LMatrix::num_partition() {
  return nPart;
}

LogicalPartition LMatrix::logical_partition() {
  return runtime->get_logical_partition(ctx, region, ipart);
}

Domain LMatrix::color_domain() {
  return color_domain;
}

void LMatrix::init
(const int nProc_, const Matrix& U_, const Matrix& V_, const Vector& D_) {

  this->nProc = nProc_;
  this->U = U_;
  this->V = V_;
  this->D = D_;  
}

// solve A x = b for each partition
//  b will be overwritten by x
void LMatrix::solve
(LMatrix& b, Context ctx, HighLevelRuntime* runtime, bool wait) {

  // check if the matrix is square
  if ( this->rows() != this->cols() )
    Error("not a square matrix!");

  solve<LeafSolveTask>(b, ctx, runtime, wait);
}

// solve the following system for every partition
// --             --  --    --     --      --
// |  I     V1'*u1 |  | eta0 |     | V1'*d1 |
// |               |  |      |  =  |        |
// | V0'*u0   I    |  | eta1 |     | V0'*d0 |
// --             --  --    --     --      --
//  VTd will be overwritten by eta
// Note VTd should have been reordered (in gemmRed)
//  as shown in the above picture.
void LMatrix::node_solve
(LMatrix& b, Context ctx, HighLevelRuntime* runtime, bool wait) {

  // pair up neighbor cells
  this->coarse_partition();
  b   ->coarse_partition();
  solve<NodeSolveTask>(b, ctx, runtime, wait);
  // recover the original partition
  b   ->fine_partition();
}

template <typename SolveTask>
void LMatrix::solve
(LMatrix& b, Context ctx, HighLevelRuntime *runtime, bool wait) {

  // A and b have the same number of partition
  assert( this->num_partition() == b.num_partition() );

  LogicalPartition APart = this->logical_partition();
  LogicalPartition bPart = b.logical_partition();
  
  Domain domain = this->color_domain();
  SolveTask launcher(domain, TaskArgument(), ArgumentMap());
  
  RegionRequirement A(APart, 0, READ_ONLY,  EXCLUSIVE, region);
  RegionRequirement b(bPart, 0, READ_WRTIE, EXCLUSIVE, region);
  A.add_field(FIDLDID_V);
  b.add_field(FIDLDID_V);
  launcher.add_region_requirement(A);
  launcher.add_region_requirement(b);
  
  FutureMap fm = runtime->execute_index_space(ctx, launcher);

  if(wait) {
    std::cout << "Wait for solve..." << std::endl;
    fm.wait_all_results();
  }
}

// compute A.transpose() * B
void LMatrix::gemmRed // static method
(double alpha, const LMatrix& A, const LMatrix& B,
 double beta,  const LMatrix& C,
 Context ctx, HighLevelRuntime *runtime, bool wait=true) {

  // A and B have the same number of partition
  assert( A.num_partition() == B.num_partition() );

  LogicalPartition APart = A.logical_partition();
  LogicalPartition BPart = B.logical_partition();
  
  Domain domain = A.color_domain();
  GemmRedTask::Args args(alpha, beta);
  TaskArgument tArgs(&args, sizeof(args));
  GemmRedTask launcher(domain, tArgs, ArgumentMap());
  
  RegionRequirement A(APart, 0,      READ_ONLY, EXCLUSIVE, region);
  RegionRequirement B(BPart, 0,      READ_ONLY, EXCLUSIVE, region);
  RegionRequirement C(BPart, SWITCH, REDOP_ADD, EXCLUSIVE, region);
  A.add_field(FIDLDID_V);
  B.add_field(FIDLDID_V);
  C.add_field(FIDLDID_V);
  launcher.add_region_requirement(A);
  launcher.add_region_requirement(B);
  launcher.add_region_requirement(C);
  
  FutureMap fm = runtime->execute_index_space(ctx, launcher);

  if(wait) {
    std::cout << "Wait for solve..." << std::endl;
    fm.wait_all_results();
  }  
}
  
