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
#if 0
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  assert(task->arglen == sizeof(TaskArgs));
  Point<1> p = task->index_point.get_point<1>();
  //printf("point = %d\n", p[0]);

  log_solver_tasks.print("Inside gemm reduction tasks.");

  const TaskArgs args = *((const TaskArgs*)task->args);
  int Arblk = args.Arblk;
  int Brblk = args.Brblk;
  int Crblk = args.Crblk;
  int Acols = args.Acols;
  int Bcols = args.Bcols;
  int Ccols = args.Ccols;
  int AcolIdx = args.AcolIdx;
  int BcolIdx = args.BcolIdx;
  int CcolIdx = args.CcolIdx;      
  //printf("A(%d, %d), B(%d, %d), C(%d, %d)\n",
  //	 Arblk, Acols, Brblk, Bcols, Crblk, Ccols);
  
  int Arlo = p[0]*Arblk;
  int Arhi = (p[0] + 1) * Arblk;
  int Brlo = p[0]*Brblk;
  int Brhi = (p[0] + 1) * Brblk;
  
  int clrSize = args.colorSize;
  int color = p[0] / clrSize;
  int Crlo = color*Crblk;
  int Crhi = (color + 1) * Crblk;
  
  PtrMatrix AMat = get_raw_pointer(regions[0], Arlo, Arhi, AcolIdx, AcolIdx+Acols);
  PtrMatrix BMat = get_raw_pointer(regions[1], Brlo, Brhi, BcolIdx, BcolIdx+Bcols);
  PtrMatrix CMat = reduction_pointer(regions[2], Crlo, Crhi, CcolIdx, CcolIdx+Ccols);
  AMat.set_trans(args.transa);
  BMat.set_trans(args.transb);
  double alpha = args.alpha;

  //printf("leading D: %d\n", CMat.LD());  
  PtrMatrix::gemm(alpha, AMat, BMat, CMat);
#endif
}

