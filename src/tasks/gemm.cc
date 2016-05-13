#include "gemm.hpp"
#include "ptr_matrix.hpp"
#include "utility.hpp"

static Realm::Logger log_solver_tasks("solver_tasks");

int GemmTask::TASKID;

GemmTask::GemmTask(TaskArgument arg,
		   Predicate pred,
		   MapperID id,
		   MappingTagID tag)  
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

void GemmTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <GemmTask::cpu_task>(AUTO_GENERATE_ID,
			 Processor::LOC_PROC, 
			 true,
			 false,
			 AUTO_GENERATE_ID,
			 TaskConfigOptions(true/*leaf*/),
			 "Gemm");

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : Gemm\n", TASKID);
#endif
}

void GemmTask::cpu_task(const Task *task,
			   const std::vector<PhysicalRegion> &regions,
			   Context ctx, HighLevelRuntime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  assert(task->arglen == sizeof(TaskArgs));

  printf("Inside gemm tasks.\n");

  const TaskArgs args = *((const TaskArgs*)task->args);
  char transA = args.transA;
  char transB = args.transB;
  double alpha = args.alpha;
  double beta  = args.beta;
  int Arows = args.Arows;
  int Brows = args.Brows;
  int Crows = args.Crows;
  int AcolIdx = args.AcolIdx;
  int BcolIdx = args.BcolIdx;
  int Acols = args.Acols;
  int Bcols = args.Bcols;
  int Ccols = args.Ccols;

  printf("Arows=%d, AcolIdx=%d, Acols=%d, "
	 "Brows=%d, BcolIdx=%d, Bcols=%d, "
	 "Crows=%d, Ccols=%d\n",
  	 Arows, AcolIdx, Acols,
	 Brows, BcolIdx, Bcols,
	 Crows, Ccols);
  
  PtrMatrix AMat = get_raw_pointer(regions[0], 0, Arows, AcolIdx, AcolIdx+Acols);
  PtrMatrix BMat = get_raw_pointer(regions[1], 0, Brows, BcolIdx, BcolIdx+Bcols);
  PtrMatrix CMat = get_raw_pointer(regions[2], 0, Crows, 0,       Ccols);
  AMat.set_trans(transA);
  BMat.set_trans(transB);

  //printf("leading D: %d\n", CMat.LD());  
  PtrMatrix::gemm(alpha, AMat, BMat, beta, CMat);
}

