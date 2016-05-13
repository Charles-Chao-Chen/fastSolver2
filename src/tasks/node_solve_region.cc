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
  TASKID = HighLevelRuntime::register_legion_task
    <NodeSolveRegionTask::cpu_task>(AUTO_GENERATE_ID,
				    Processor::LOC_PROC, 
				    true,
				    false,
				    AUTO_GENERATE_ID,
				    TaskConfigOptions(true/*leaf*/),
				    "Node_Solve_Region");

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
			     Context ctx, HighLevelRuntime *runtime) {

  printf("Inside node solve tasks.\n");

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
      
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  assert(task->arglen == sizeof(TaskArgs));

  const TaskArgs args = *((const TaskArgs*)task->args);
  int rank = args.rank;
  int nRhs = args.nRhs;
  printf("rank=%d, nRhs=%d\n", rank, nRhs);

  PtrMatrix VTu0 = get_raw_pointer(regions[0], 0, rank, 0, rank);
  PtrMatrix VTu1 = get_raw_pointer(regions[1], 0, rank, 0, rank);
  //PtrMatrix VTd0 = get_raw_pointer(regions[2], 0, rank, 0, nRhs);
  //PtrMatrix VTd1 = get_raw_pointer(regions[3], 0, rank, 0, nRhs);
 
#if 0

  PtrMatrix S(2*rank, 2*rank);
  PtrMatrix B(2*rank, nRhs);
  
  // assume V0'*u0 and V1'*u1 have the same number of rows
  S.identity(); // initialize to identity matrix
  int r = rank;
  for (int i=0; i<r; i++) {
    for (int j=0; j<r; j++) {
      S(r+i, j) = VTu0(i, j);
      S(i, r+j) = VTu1(i, j);
    }
  }
  // set the right hand side
  for (int j=0; j<nRhs; j++) {
    for (int i=0; i<r; i++) {
      B(i,   j) = VTd1(i, j);
      B(i+r, j) = VTd0(i, j);
    }
  }  
  S.solve( B );
#endif
}
