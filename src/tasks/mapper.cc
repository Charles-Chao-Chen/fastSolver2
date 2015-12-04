#include "legion.h"
#include "utilities.h"
#include "mapper.hpp"

Realm::Logger log_solver_mapper("solver_mapper");

SolverMapper::SolverMapper
(Machine m, HighLevelRuntime *rt, Processor p)
  : DefaultMapper(m, rt, p) {

  // select and store all SYSTEM_MEM's in valid_mems
  std::set<Memory> all_mems;
  machine.get_all_memories(all_mems);
  std::set<Memory>::const_iterator it_mem = all_mems.begin();
  for (; it_mem != all_mems.end(); it_mem++) {
    if (it_mem->kind() == Memory::SYSTEM_MEM) {
      this->valid_mems.push_back(*it_mem);
      
      // select valid processors for this memory
      std::vector<Processor> valid_procs;
      std::set<Processor> all_procs;
      machine.get_shared_processors(*it_mem, all_procs);
      std::set<Processor>::const_iterator it_proc = all_procs.begin();
      for (; it_proc != all_procs.end(); it_proc++) {
	if ( it_proc->kind() == Processor::LOC_PROC)
	  valid_procs.push_back(*it_proc);
      }

      // map this memory to correspoind processors
      mem_procs[*it_mem] = valid_procs;
    }    
  }
  this->num_mems = valid_mems.size();
  assert( ! valid_mems.empty() );

  // print machine information
  if ( local_proc == mem_procs[valid_mems[0]][0] ) {
    std::cout << "There are " << valid_mems.size()
	      << " machines" << std::endl;
    std::cout << "Every machine has "
	      << mem_procs[ valid_mems[0] ].size()
	      << " cores." << std::endl;
#ifdef DEBUG_SOLVER_MAPPER
    std::cout << "Use the first " << num_mems-1 << " machines for"
      " running tasks and the last one for top-level task.\n";
#endif
  }
#ifdef DEBUG_SOLVER_MAPPER
  num_mems--;
#endif
}

void SolverMapper::select_task_options(Task *task) {

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

#if 0
// assign chunk of points to different processors
void SolverMapper::slice_domain(const Task *task, const Domain &domain,
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
    
  int nbucket;
  int bucket_size;
  if (num_elmts < num_mems) {
    assert(num_mems % num_elmts == 0);
    nbucket = num_elmts;
    bucket_size = num_mems / num_elmts;
    for (int i=0; i<nbucket; i++) {
      Point<1> lo(i);
      Point<1> hi(i);
      Rect<1> chunk(lo, hi);
      int mem_idx = i*bucket_size;
      Processor target = mem_procs[valid_mems[mem_idx]][0];
      DomainSplit ds(Domain::from_rect<1>(chunk), target, false, false);
      slices.push_back(ds);

      /*
      std::cout << "domain (" << lo.x[0] << ", " << hi.x[0] << ")"
		<< " is assigned to machine: "
		<< mem_idx << std::endl;
      */
    }
  }

  if (num_elmts >= num_mems) {
    assert(num_elmts % num_mems == 0);
    nbucket = num_mems;
    bucket_size = num_elmts / num_mems;
    for (int i=0; i<nbucket; i++) {
      Point<1> lo(i*bucket_size);
      Point<1> hi((i+1)*bucket_size-1);
      Rect<1> chunk(lo, hi);
      int mem_idx = i;
      Processor target = mem_procs[valid_mems[mem_idx]][0];
      DomainSplit ds(Domain::from_rect<1>(chunk), target, false, false);
      slices.push_back(ds);

      /*
      std::cout << "domain (" << lo.x[0] << ", " << hi.x[0] << ")"
		<< " is assigned to machine: "
		<< mem_idx << std::endl;
      */
    }
  }
}

#else
// assign every point in the launch domain to different processors
void SolverMapper::slice_domain(const Task *task, const Domain &domain,
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

    /*
  int lower = num_elmts / num_mems;
  int upper = lower + 1;
  int num_upper = num_elmts % num_mems;
  int num_lower = num_mems - num_upper;

  int mem_idx = 0;
  for (int i=0; i<num_elmts; i++) {
    Point<1> lo(i);
    Point<1> hi(i);
    Rect<1> chunk(lo, hi);
    //int mem_idx = i * num_mems / num_elmts;
    
    
    Processor target = mem_procs[valid_mems[mem_idx]][0];
    DomainSplit ds(Domain::from_rect<1>(chunk), target, false, false);
    slices.push_back(ds);
    log_solver_mapper.print("task %s: Point(%i) is assigned to"
			    "machine: %i", task->variants->name, i, mem_idx);
  }
  */
}
#endif

/*
void SolverMapper::slice_domain(const Task *task, const Domain &domain,
				std::vector<DomainSplit> &slices) {
  
#if 0
  std::cout << "inside slice_domain()" << std::endl;
  std::cout << "orign: " << task->orig_proc.id
	    << ", current: " << task->current_proc.id
	    << ", target: " << task->target_proc.id
	    << std::endl;
#endif

  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);
  std::vector<Processor> split_set;
  for (unsigned idx = 0; idx < 2; idx++)
  {
    split_set.push_back(DefaultMapper::select_random_processor(
                        all_procs, Processor::LOC_PROC, machine));
  }

  DefaultMapper::decompose_index_space(domain, split_set, 
                                        1, slices);

  for (std::vector<DomainSplit>::iterator it = slices.begin();
        it != slices.end(); it++)
  {
    Rect<1> rect = it->domain.get_rect<1>();
    if (rect.volume() == 1) {
      it->recurse = false;
      int num_elmts = task->tag;
      int mem_idx = rect.lo.x[0] * num_mems / num_elmts;
      it->proc = mem_procs[valid_mems[mem_idx]][0];
    }
    else
      it->recurse = true;
  }
}
*/

bool SolverMapper::map_task(Task *task) {
  
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

void SolverMapper::notify_mapping_failed(const Mappable *mappable)
{
  printf("WARNING: MAPPING FAILED!  Retrying...\n");
}
/*
void SolverMapper::notify_mapping_result(const Mappable *mappable)
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
