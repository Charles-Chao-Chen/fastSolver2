#include "gemm_reduce.hpp"

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
			    "GemmRed_Solve");

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : GemmRed_Solve\n", TASKID);
#endif
}

void GemmRedTask::cpu_task(const Task *task,
			   const std::vector<PhysicalRegion> &regions,
			   Context ctx, HighLevelRuntime *runtime) {
  
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  assert(task->arglen == sizeof(TaskArgs));
  /*
  Point<1> p = task->index_point.get_point<1>();
  printf("point = %d\n", p[0]);

  const TaskArgs args = *((const TaskArgs*)task->args);
  int rblk  = args.nrow;
  int nRhs  = args.nRhs;

  int rlo = p[0]*rblk;
  int rhi = (p[0] + 1) * rblk;
  double *Aptr = region_pointer(regions[0], rlo, rhi, 0, A.cols);
  double *Bptr = region_pointer(regions[1], rlo, rhi, 0, B.cols);
  double *Cptr = region_pointer(regions[2], C.rlo, C.rhi, 0, nRhs);
  
  PtrMatrix AMat(rblk, rblk, rblk, Aptr);
  PtrMatrix BMat(rblk, nRhs, rblk, Bptr);
  PtrMatrix CMat(rblk, nRhs, rblk, Cptr);
  //AMat.solve( BMat );
  PtrMatrix::gemm(AMat, BMat, CMat);


  PtrMatrix AMat = get_raw_pointer(regions[0], rlo, rhi, 0, A.cols);
  PtrMatrix BMat = get_raw_pointer(regions[1], rlo, rhi, 0, B.cols);
  PtrMatrix CMat = get_raw_pointer(regions[2], C.rlo, C.rhi, 0, C.cols);
  AMat.set_trans('t');
  PtrMatrix::gemm(alpha, AMat, BMat, beta, CMat);
  */
}

