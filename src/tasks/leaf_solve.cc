#include "leaf_solve.hpp"
#include "ptr_matrix.hpp"
#include "utility.hpp"
#include <math.h>

static Realm::Logger log_solver_tasks("solver_tasks");

void hsolve
(int nrow, int nrhs, int rank, int nPart, int LD,
 double *K, double *U, double *V);
  
int LeafSolveTask::TASKID;

LeafSolveTask::LeafSolveTask(Domain domain,
			     TaskArgument global_arg,
			     ArgumentMap arg_map,
			     MappingTagID tag,
			     Predicate pred,
			     bool must,
			     MapperID id)
  
  : IndexLauncher(TASKID, domain, global_arg,
		  arg_map, pred, must, id, tag) {}

void LeafSolveTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <LeafSolveTask::cpu_task>(AUTO_GENERATE_ID,
			    Processor::LOC_PROC, 
			    false,
			    true,
			    AUTO_GENERATE_ID,
			    TaskConfigOptions(true/*leaf*/),
			    "Leaf_Solve");

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : Leaf_Solve\n", TASKID);
#endif
}

void LeafSolveTask::cpu_task(const Task *task,
			     const std::vector<PhysicalRegion> &regions,
			     Context ctx, HighLevelRuntime *runtime) {

  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  assert(task->arglen == sizeof(TaskArgs));
  Point<1> p = task->index_point.get_point<1>();  
  log_solver_tasks.print("Inside leaf solve tasks.");

  const TaskArgs args = *((const TaskArgs*)task->args);
  int rblk  = args.nrow;
  int nRhs  = args.nRhs;
  int rank  = args.rank;
  int nPart = args.nPart;
  int level = log2(nPart);
  //assert(rank*nPart==rblk);
  int rlo = p[0]*rblk;
  int rhi = (p[0] + 1) * rblk;
  PtrMatrix KMat = get_raw_pointer(regions[0], rlo, rhi, 0, rblk/nPart);
  PtrMatrix UMat = get_raw_pointer(regions[1], rlo, rhi, 0, nRhs);
  PtrMatrix VMat = get_raw_pointer(regions[2], rlo, rhi, 0, rank);
  assert(KMat.LD() == UMat.LD());
  assert(KMat.LD() == VMat.LD());
  //std::cout<<"nPart:"<<nPart<<", level:"<<level<<std::endl;
  assert(nPart==(int)pow(2,level));
#ifdef DEBUG_SOLVER
  std::cout<<"point:"<<p[0]<<std::endl;
  std::cout<<"nrow:"<<rblk<<", nRhs:"<<nRhs<<", rank:"<<rank
	   <<", nPart:"<<nPart<<", LD:"<<KMat.LD()<<std::endl;
#endif
  hsolve(rblk, nRhs-level*rank, rank, nPart, KMat.LD(),
  	 KMat.pointer(), UMat.pointer(), VMat.pointer());
}

void hsolve
(int nrow, int nrhs, int rank, int nPart, int LD,
 double *K, double *U, double *V) {
#ifdef DEBUG_SOLVER
  std::cout<<"nrow:"<<nrow<<", nRhs:"<<nrhs<<", rank:"<<rank
	   <<", nPart:"<<nPart<<", LD:"<<LD<<std::endl;
#endif
  if (nPart==1) {
    int     N    = nrow;
    int     NRHS = nrhs;
    int     LDA  = LD;
    int     LDB  = LD;
    double *A    = K;
    double *B    = U;
    int     INFO;
    int     IPIV[N];
    lapack::dgesv7_(&N, &NRHS, A, &LDA, IPIV, B, &LDB, &INFO);
    assert(INFO == 0);
    return;
  }

  // recursively solve two children
  assert(nrow%2==0);
  assert(nPart%2==0);
  double *d0 = U;
  double *d1 = U  + nrow/2;
  double *V0 = V;
  double *V1 = V  + nrow/2;
  double *u0 = d0 + nrhs*LD;
  double *u1 = d1 + nrhs*LD;
  hsolve(nrow/2, nrhs+rank, rank, nPart/2, LD, K,        d0, V0);
  hsolve(nrow/2, nrhs+rank, rank, nPart/2, LD, K+nrow/2, d1, V1);

  char   transa = 't';
  char   transb = 'n';
  double alpha  = 1.0;
  double beta   = 0.0;

  int V0_rows = nrow/2, V1_rows = nrow/2;
  int V0_cols = rank,   V1_cols = rank;
  int u0_rows = nrow/2, u1_rows = nrow/2;
  int u0_cols = rank,   u1_cols = rank;
  //int d0_rows = nrow, d1_rows = nrow;
  int d0_cols = nrhs,   d1_cols = nrhs;

  // form the Schur complement, refer to the algorithm in HMatrix.cc
  int     S_size = 2*rank;
  double *S   = (double *) calloc(S_size * S_size, sizeof(double));
  double *RHS = (double *) malloc(S_size * nrhs * sizeof(double));
  for (int i=0; i<S_size; i++) {
    S[S_size*i+i] = 1.0;
  }
  
  // set the pointers
  double *V0Tu0 = S + S_size/2;
  double *V1Tu1 = S + S_size/2*S_size;
  double *V0Td0 = RHS + S_size/2;
  double *V1Td1 = RHS;
  
  blas::dgemm7_(&transa, &transb, &V0_cols, &u0_cols, &V0_rows, &alpha, V0, &LD, u0, &LD, &beta, V0Tu0, &S_size);
  blas::dgemm7_(&transa, &transb, &V1_cols, &u1_cols, &V1_rows, &alpha, V1, &LD, u1, &LD, &beta, V1Tu1, &S_size);
  blas::dgemm7_(&transa, &transb, &V0_cols, &d0_cols, &V0_rows, &alpha, V0, &LD, d0, &LD, &beta, V0Td0, &S_size);
  blas::dgemm7_(&transa, &transb, &V1_cols, &d1_cols, &V1_rows, &alpha, V1, &LD, d1, &LD, &beta, V1Td1, &S_size);

  int INFO;
  int IPIV[S_size];
  assert(d0_cols == d1_cols);
  lapack::dgesv7_(&S_size, &d0_cols, S, &S_size, IPIV, RHS, &S_size, &INFO);
  assert(INFO == 0);

  transa =  'n';
  alpha  = -1.0;
  beta   =  1.0;

  //int eta0_rows = S_size/2, eta1_rows = S_size/2;
  int eta0_cols = d0_cols,  eta1_cols = d0_cols;
  double *eta0 = V1Td1;
  double *eta1 = V0Td0;
  
  blas::dgemm7_(&transa, &transb, &u0_rows, &eta0_cols, &u0_cols, &alpha, u0, &LD, eta0, &S_size, &beta, d0, &LD);
  blas::dgemm7_(&transa, &transb, &u1_rows, &eta1_cols, &u1_cols, &alpha, u1, &LD, eta1, &S_size, &beta, d1, &LD);
}

