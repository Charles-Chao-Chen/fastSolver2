#include "dist_mapper.hpp"

DistMapper::DistMapper
(Machine m, Runtime *rt, Processor p, int radix_)
  : DefaultMapper(m, rt, p) {

  this->radix = radix_;
  
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
    for (size_t i=0; i<valid_mems.size(); i++) {
      std::cout << "Machine " << i << " has "
		<< mem_procs[ valid_mems[i] ].size()
		<< " cores." << std::endl;
    }
  }
}

void DistMapper::select_task_options(Task *task) {

  // only top level task is not index space task
  if (! task->is_index_space) {
  
    task->inline_task   = false;
    task->spawn_task    = false;
    task->map_locally   = false;
    task->profile_task  = false;
    task->task_priority = 0;

    // top level task run on machine 0
    std::vector<Processor> procs = mem_procs[ valid_mems[0] ];
    assert( !procs.empty() );  
    task->target_proc = procs[0];
  }

  // index space tasks
  else {
    //std::cout << "index space task" << std::endl;
    task->inline_task   = false;
    task->spawn_task    = false;
    task->map_locally   = false; // remote mapping
    task->profile_task  = false;
    task->task_priority = 0;
  
    std::vector<Processor> procs = mem_procs[ valid_mems[0] ];
    assert( !procs.empty() );
    task->target_proc = procs[0];

    // task->tag stores the size of launch domain
    assert(task->tag > 0);

    int num_tasks = task->tag;
    //std::cout << "domain size: " << num_tasks << std::endl;
    if (num_tasks > num_mems)
      // assume the same number of tasks on every machine
      assert(num_tasks % num_mems == 0);
    else
      assert(num_mems % num_tasks == 0);
  }  
}

void DistMapper::slice_domain(const Task *task, const Domain &domain,
			      std::vector<DomainSplit> &slices) {
  
#if 0
  std::cout << "inside slice_domain()" << std::endl;
  std::cout << "orign: " << task->orig_proc.id
	    << ", current: " << task->current_proc.id
	    << ", target: " << task->target_proc.id
	    << std::endl;
#endif

  int num_tasks = task->tag;

  // the number of node solve tasks decreases from leaf to root,
  // which may be smaller than the number of machines
  if (num_tasks <= num_mems) {
    //std::cout << "There are " << num_tasks << " tasks" << std::endl;
    // send every point to the target machine
    for (int i=0; i<num_tasks; i++) {
      int mem_idx = i * num_mems / num_tasks;
      Processor target = mem_procs[valid_mems[mem_idx]][0];
      Point<1> lo(i), hi(i);
      Rect<1> chunk(lo, hi);
      DomainSplit ds(Domain::from_rect<1>(chunk), target, false, false);
      slices.push_back(ds);
      //std::cout << "Point: " << i << " is assigned to machine: "
      //	<< mem_idx << std::endl;
    }
    return;
  }
  
  // when the number of tasks is more than the number of machines
  int bucket_size = num_tasks / num_mems;
  
  // current task domain
  Rect<1> rect = domain.get_rect<1>();
  int num_elmts = rect.volume();
  int half_domain = num_elmts / 2;
  bool recurse = half_domain < 2*bucket_size ? false : true;
  assert(num_elmts % 2 == 0);
  
  // slice domain into two pieces
  for (int i=0; i<2; i++) {
    
    // extract the sub-domain
    Point<1> lo(rect.lo.x[0]+i*half_domain);
    Point<1> hi(rect.lo.x[0]+(i+1)*half_domain-1);
    Domain dom = Domain::from_rect<1>(Rect<1>(lo, hi));
    
    // find the target machine
    int mem_idx = lo.x[0] / bucket_size;
    Processor target = mem_procs[valid_mems[mem_idx]][0];

    // create a domain slice
    DomainSplit ds(dom, target, recurse, false);
    slices.push_back(ds);
    /*
    std::cout << "domain (" << lo.x[0] << ", " << hi.x[0] << ")"
    	      << " is assigned to machine: "
    	      << mem_idx << std::endl;
    */
  }
}

bool DistMapper::map_task(Task *task) {

#if 0
  std::cout << "Inside map_task() ..." << std::endl;
  std::cout << "orign: " << task->orig_proc.id
	    << ", current: " << task->current_proc.id
	    << ", target: " << task->target_proc.id
	    << std::endl;
#endif
  
  // find the memory associated with the target processor
  Memory sys_mem = machine_interface.find_memory_kind
    (task->target_proc, Memory::SYSTEM_MEM); 
  assert(sys_mem.exists());

  // assign additional processors
  std::vector<Processor>& procs = mem_procs[sys_mem];
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

void DistMapper::notify_mapping_failed(const Mappable *mappable)
{
  printf("WARNING: MAPPING FAILED!  Retrying...\n");
}
/*
void DistMapper::notify_mapping_result(const Mappable *mappable)
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
