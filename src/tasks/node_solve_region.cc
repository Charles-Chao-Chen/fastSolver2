#include "node_solve_region.hpp"
#include "ptr_matrix.hpp"
#include "utility.hpp"

static Realm::Logger log_solver_tasks("solver_tasks");

int NodeSolveRegionTask::TASKID;

NodeSolveRegionTask::NodeSolveRegionTask(TaskArgument arg,
					 Predicate pred,
					 MapperID id,
					 MappingTagID tag)  
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

void NodeSolveRegionTask::register_tasks(void)
{
  TASKID = Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(TASKID, "Node_Solve_Region");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_leaf(true);
  Runtime::preregister_task_variant<NodeSolveRegionTask::cpu_task>(registrar, "cpu");

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : Node_Solve_Region\n", TASKID);
#endif
}

// solve the following system
// --             --  --    --     --      --
// |  I     V1'*u1 |  | eta0 |     | V1'*d1 |
// |               |  |      |  =  |        |
// | V0'*u0   I    |  | eta1 |     | V0'*d0 |
// --             --  --    --     --      --
// note the reverse order in VTd
void NodeSolveRegionTask::cpu_task(const Task *task,
			     const std::vector<PhysicalRegion> &regions,
			     Context ctx, Runtime *runtime) {

  //printf("Inside node solve tasks.\n");
#if 0
  LogicalRegion VTu0_rg = regions[0].get_logical_region();
  LogicalRegion VTu1_rg = regions[1].get_logical_region();
  LogicalRegion VTd0_rg = regions[2].get_logical_region();
  LogicalRegion VTd1_rg = regions[3].get_logical_region();
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
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  assert(task->arglen == sizeof(TaskArgs));

  const TaskArgs args = *((const TaskArgs*)task->args);
  int rank0 = args.rank0;
  int rank1 = args.rank1;
  int nRhs  = args.nRhs;
  //printf("rank=%d, nRhs=%d\n", rank, nRhs);

  PtrMatrix VTu0 = get_raw_pointer(regions[0], 0, rank1, 0, rank0);
  PtrMatrix VTu1 = get_raw_pointer(regions[1], 0, rank0, 0, rank1);
  PtrMatrix VTd0 = get_raw_pointer(regions[2], 0, rank1, 0, nRhs);
  PtrMatrix VTd1 = get_raw_pointer(regions[3], 0, rank0, 0, nRhs);
 
  int N = rank0 + rank1;
  PtrMatrix S(N, N);
  PtrMatrix B(N, nRhs);
  
  S.identity(); // initialize to identity matrix
  for (int i=0; i<rank0; i++) {
    for (int j=0; j<rank1; j++) {
      S(i, rank0+j) = VTu1(i, j);
      S(rank0+j, i) = VTu0(j, i);
    }
  }
  // set the right hand side
  for (int j=0; j<nRhs; j++) {
    for (int i=0; i<rank0; i++) {
      B(i,   j) = VTd1(i, j);
    }
    for (int i=0; i<rank1; i++) {
      B(rank0+i, j) = VTd0(i, j);
    }
  }  
  S.solve( B );
  
  // copy back the results
  for (int j=0; j<nRhs; j++) {
    for (int i=0; i<rank0; i++) {
      VTd1(i, j) = B(i, j);
    }
    for (int i=0; i<rank1; i++) {
      VTd0(i, j) = B(rank0+i, j);
    }
  }    
}
