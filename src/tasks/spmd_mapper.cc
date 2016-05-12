#include "legion.h"
#include "utilities.h"
#include "smpd_mapper.hpp"

Realm::Logger log_solver_mapper("solver_mapper");

SPMDsolverMapper::SPMDsolverMapper(Machine machine, HighLevelRuntime *rt, Processor local,
				   std::vector<Processor>* _procs_list,
				   std::vector<Memory>* _sysmems_list,
				   std::map<Memory, std::vector<Processor> >* _sysmem_local_procs,
				   std::map<Processor, Memory>* _proc_sysmems,
				   std::map<Processor, Memory>* _proc_regmems)
  : DefaultMapper(machine, rt, local),
    procs_list(*_procs_list),
    sysmems_list(*_sysmems_list),
    sysmem_local_procs(*_sysmem_local_procs),
    proc_sysmems(*_proc_sysmems),
    proc_regmems(*_proc_regmems)
{}

void SPMDsolverMapper::select_task_options(Task *task) {

  // only top level task is not index space task
  if (! task->is_index_space) {
  
    task->inline_task   = false;
    task->spawn_task    = false;
    task->map_locally   = false;
    task->profile_task  = false;
    task->task_priority = 0;

    // top level task run on machine 0
#ifdef DEBUG_SOLVER_MAPPER
    std::vector<Processor> procs = mem_procs[ valid_mems[num_mems] ];
#else
    std::vector<Processor> procs = mem_procs[ valid_mems[0] ];
#endif
    assert( !procs.empty() );  
    task->target_proc = procs[0];
  }

  // index space tasks
  else {
    //std::cout << "index space task" << std::endl;
    task->inline_task   = false;
    task->spawn_task    = false;
    task->map_locally   = true;
    task->profile_task  = false;
    task->task_priority = 0;
  
    // assign a dummy processor
    std::vector<Processor> procs = mem_procs[ valid_mems[0] ];
    //std::vector<Processor> procs = mem_procs[ valid_mems[num_mems] ];
    assert( !procs.empty() );
    task->target_proc = procs[0];
  }  
}

// assign every point in the launch domain to different processors
void SPMDsolverMapper::slice_domain(const Task *task, const Domain &domain,
				std::vector<DomainSplit> &slices) {
  
#if 0
  std::cout << "inside slice_domain()" << std::endl;
  std::cout << "orign: " << task->orig_proc.id
	    << ", current: " << task->current_proc.id
	    << ", target: " << task->target_proc.id
	    << std::endl;
#endif

  assert(domain.get_dim() == 1);
  Rect<1> rect = domain.get_rect<1>();
  int num_elmts = rect.volume();

  /*
  // assume evenly split
  if (num_elmts > num_mems) 
    assert(num_elmts % num_mems == 0);
  else
    assert(num_mems % num_elmts == 0);
    // if num_mems > num_elmts, e.g.,
    //  num_mems = 8, num_elmts = 4,
    //  the assignment is {0, 2, 4, 6}
    // if num_mems < num_elmts, e.g.,
    //  num_mems = 4, num_elmts = 8,
    //  the assignment is {0, 0, 1, 1, ..., 4, 4}
    */

  if (num_elmts >= num_mems) {
    // e.g. num_elmts = 64x8, num_mems = 62
    // lower = 8, upper = 9
    // num_lower = 46, num_upper = 16
    int lower = num_elmts / num_mems;
    int upper = lower + 1;
    int num_upper = num_elmts % num_mems;
    int num_lower = num_mems - num_upper;
    int mem_idx = 0;
    int num_task = 0;
    for (int i=0; i<num_elmts; i++) {
      Point<1> lo(i);
      Point<1> hi(i);
      Rect<1> chunk(lo, hi);
      // compute machine index
      if (mem_idx < num_lower) {
	if (num_task < lower) {
	  num_task++;
	}
	else {
	  mem_idx++;
	  num_task=1;
	}
      }
      else { // mem_idx >= num_lower
	if (num_task < upper) {
	  num_task++;
	}
	else {
	  mem_idx++;
	  num_task=1;
	}
      }
      //end if
      assert(mem_idx < num_mems);
      Processor target = mem_procs[valid_mems[mem_idx]][0];
      DomainSplit ds(Domain::from_rect<1>(chunk), target, false, false);
      slices.push_back(ds);
      log_solver_mapper.print("task %s: Point(%i) is assigned to "
			      "machine: %i", task->variants->name, i,
			      mem_idx);
    }
  }
    
  else { // num_elmts < num_mems
    // different lower and upper definitions here
    // e.g. num_mems = 62, num_elmts = 8
    // lower = 7, upper = 8
    // num_lower = 2, num_upper = 6
    // the result is: [0,8,16,...,48,55]
    int lower = num_mems / num_elmts;
    int upper = lower + 1;
    int num_upper = num_mems % num_elmts;
    int mem_idx = 0;
    for (int i=0; i<num_elmts; i++) {
      Point<1> lo(i);
      Point<1> hi(i);
      Rect<1> chunk(lo, hi);
      assert(mem_idx < num_mems);
      Processor target = mem_procs[valid_mems[mem_idx]][0];
      DomainSplit ds(Domain::from_rect<1>(chunk), target, false, false);
      slices.push_back(ds);
      log_solver_mapper.print("task %s: Point(%i) is assigned to "
			      "machine: %i", task->variants->name, i,
			      mem_idx);
      // compute next machine index
      if (i < num_upper) {
	mem_idx += upper;
      }
      else {
	mem_idx += lower;
      }
      // end if
    }
  }
}

bool SPMDsolverMapper::map_task(Task *task) {
  
#if 0
  std::cout << "Inside map_task() ..." << std::endl;
  std::cout << "orign: " << task->orig_proc.id
	    << ", current: " << task->current_proc.id
	    << ", target: " << task->target_proc.id
	    << std::endl;
#endif

  log_solver_mapper.print("map task %s: %i", task->variants->name, task->index_point.point_data[0]);
  
  // find the memory associated with the target processor
  Memory sys_mem = machine_interface.find_memory_kind
    (task->target_proc, Memory::SYSTEM_MEM); 
  assert(sys_mem.exists());

  // assign additional processors
  std::vector<Processor>& procs = mem_procs[sys_mem];
  assert(!procs.empty());
  task->additional_procs.insert(procs.begin(), procs.end());

  // map the regions
  for (unsigned idx = 0; idx < task->regions.size(); idx++) {
    task->regions[idx].target_ranking.push_back(sys_mem);
    task->regions[idx].virtual_map = false;
    task->regions[idx].enable_WAR_optimization = war_enabled;
    task->regions[idx].reduction_list = false;
    task->regions[idx].blocking_factor = 1;
  } 
  return true;
}

void SPMDsolverMapper::notify_mapping_failed(const Mappable *mappable)
{
  printf("WARNING: MAPPING FAILED!  Retrying...\n");
}
/*
void SPMDsolverMapper::notify_mapping_result(const Mappable *mappable)
{
  std::cout << "inside notify_result()" << std::endl;
    
#if 0
  if (mappable->get_mappable_kind() == Mappable::TASK_MAPPABLE)
    {
      const Task *task = mappable->as_mappable_task();
      assert(task != NULL);
      for (unsigned idx = 0; idx < task->regions.size(); idx++)
	{
	  printf("Mapped region %d of task %s (ID %lld) to memory %x\n",
		 idx, task->variants->name, 
		 task->get_unique_task_id(),
		 task->regions[idx].selected_memory.id);
	}
    }
#endif
}
*/
