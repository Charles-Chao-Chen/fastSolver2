#include "gemm_broadcast.hpp"
#include "ptr_matrix.hpp"
#include "utility.hpp"

static Realm::Logger log_solver_tasks("solver_tasks");

int GemmBroTask::TASKID;

GemmBroTask::GemmBroTask(Domain domain,
			 TaskArgument global_arg,
			 ArgumentMap arg_map,
			 MappingTagID tag,
			 Predicate pred,
			 bool must,
			 MapperID id)
  
  : IndexLauncher(TASKID, domain, global_arg,
		  arg_map, pred, must, id, tag) {}

void GemmBroTask::register_tasks(void)
{
  TASKID = Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(TASKID, "Gemm_Bro");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_leaf(true);
  Runtime::preregister_task_variant<GemmBroTask::cpu_task>(registrar, "cpu");

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : GemmBroadcast\n", TASKID);
#endif
}

void GemmBroTask::cpu_task(const Task *task,
			   const std::vector<PhysicalRegion> &regions,
			   Context ctx, Runtime *runtime) {

  //assert(regions.size() == 3);
  //assert(task->regions.size() == 3);
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(TaskArgs));
  Point<1> p(task->index_point);
  //printf("point = %d\n", p[0]);

  log_solver_tasks.print("Inside gemm broadcast tasks.");

  const TaskArgs args = *((const TaskArgs*)task->args);
  int Arblk = args.Arblk;
  int Brblk = args.Brblk;
  int Crblk = args.Crblk;
  int Acols = args.Acols;
  int Bcols = args.Bcols;
  int Ccols = args.Ccols;
  int AcolIdx = args.AcolIdx;
  //printf("A(%d, %d), B(%d, %d), C(%d, %d)\n",
  //	 Arblk, Acols, Brblk, Bcols, Crblk, Ccols);
  
  int Arlo = p[0]*Arblk;
  int Arhi = (p[0] + 1) * Arblk;
  int Crlo = p[0]*Crblk;
  int Crhi = (p[0] + 1) * Crblk;
  
  int clrSize = args.colorSize;
  int color = p[0] / clrSize;
  int Brlo = color*Brblk;
  int Brhi = (color + 1) * Brblk;
  
  PtrMatrix AMat = get_raw_pointer<LEGION_READ_WRITE>(regions[0], Arlo, Arhi, AcolIdx, AcolIdx+Acols);
  PtrMatrix BMat = get_raw_pointer<LEGION_READ_ONLY>(regions[1], Brlo, Brhi, 0, Bcols);
  //PtrMatrix CMat = get_raw_pointer<LEGION_READ_WRITE>(regions[2], Crlo, Crhi, 0, Ccols);
  PtrMatrix CMat = get_raw_pointer<LEGION_READ_WRITE>(regions[0], Crlo, Crhi, 0, Ccols);
  AMat.set_trans(args.transa);
  BMat.set_trans(args.transb);
  double alpha = args.alpha;

  /*
  std::cout << "gemm:" << std::endl;
  AMat.display("A");
  BMat.display("B");
  CMat.display("C");
*/
  PtrMatrix::gemm(alpha, AMat, BMat, CMat);
}

