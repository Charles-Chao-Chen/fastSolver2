#include "gemm_inplace.hpp"
#include "ptr_matrix.hpp"
#include "utility.hpp"

static Realm::Logger log_solver_tasks("solver_tasks");

int GemmInplaceTask::TASKID;

GemmInplaceTask::GemmInplaceTask
(TaskArgument arg, Predicate pred, MapperID id, MappingTagID tag)  
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

void GemmInplaceTask::register_tasks(void)
{
  TASKID = Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(TASKID, "Gemm_Inplace");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_leaf(true);
  Runtime::preregister_task_variant<GemmInplaceTask::cpu_task>(registrar, "cpu");

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : GemmInplace\n", TASKID);
#endif
}

void GemmInplaceTask::cpu_task(const Task *task,
			   const std::vector<PhysicalRegion> &regions,
			   Context ctx, Runtime *runtime) {
  
  //printf("Inside gemm inplace tasks.\n");

  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(TaskArgs));

  const TaskArgs args = *((const TaskArgs*)task->args);
  char transA = args.transA;
  char transB = args.transB;
  double alpha = args.alpha;
  double beta  = args.beta;
  int Arows = args.Arows;
  int Brows = args.Brows;
  int Crows = args.Crows;
  int Acols = args.Acols;
  int Bcols = args.Bcols;
  int Ccols = args.Ccols;
  int AcolIdx = args.AcolIdx;
#if 0
  printf("Arows=%d, AcolIdx=%d, Acols=%d\n"
	 "Brows=%d, Bcols=%d\n"
	 "Crows=%d, Ccols=%d\n",
  	 Arows, AcolIdx, Acols,
	 Brows, Bcols,
	 Crows, Ccols);
#endif
  PtrMatrix AMat = get_raw_pointer<LEGION_READ_WRITE>(regions[0], 0, Arows, AcolIdx, AcolIdx+Acols);
  PtrMatrix BMat = get_raw_pointer<LEGION_READ_WRITE>(regions[1], 0, Brows, 0, Bcols);
  PtrMatrix CMat = get_raw_pointer<LEGION_READ_WRITE>(regions[0], 0, Crows, 0, Ccols);
  AMat.set_trans(transA);
  BMat.set_trans(transB);

  /*
  std::cout << "gemm:" << std::endl;
  AMat.display("A");
  BMat.display("B");
  CMat.display("C");
*/
  PtrMatrix::gemm(alpha, AMat, BMat, beta, CMat);
}

