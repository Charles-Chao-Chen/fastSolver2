#include "gemm_reduce.hpp"
#include "ptr_matrix.hpp"
#include "utility.hpp"

int GemmRedTask::TASKID;

GemmRedTask::GemmRedTask(Domain domain,
			 TaskArgument global_arg,
			 ArgumentMap arg_map,
			 Predicate pred,
			 bool must,
			 MapperID id,
			 MappingTagID tag)
  
  : IndexLauncher(TASKID, domain, global_arg,
		  arg_map, pred, must, id, tag) {}

void GemmRedTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <GemmRedTask::cpu_task>(AUTO_GENERATE_ID,
			    Processor::LOC_PROC, 
			    false,
			    true,
			    AUTO_GENERATE_ID,
			    TaskConfigOptions(true/*leaf*/),
			    "GemmRed");

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : GemmRed\n", TASKID);
#endif
}

void GemmRedTask::cpu_task(const Task *task,
			   const std::vector<PhysicalRegion> &regions,
			   Context ctx, HighLevelRuntime *runtime) {
  
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  assert(task->arglen == sizeof(TaskArgs));
  Point<1> p = task->index_point.get_point<1>();
  //printf("point = %d\n", p[0]);

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
  /*
  std::cout << "gemm:" << std::endl;
  AMat.display("A");
  BMat.display("B");
  CMat.display("C");
*/
}

