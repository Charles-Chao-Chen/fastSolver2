#include "mapper.hpp"

SolverMapper::SolverMapper
(Machine m, HighLevelRuntime *rt, Processor p)
  : DefaultMapper(m, rt, p) {

  // select valid memories
  std::set<Memory> all_mems;
  machine.get_all_memories(all_mems);
  std::set<Memory>::const_iterator it_mem = all_mems.begin();
  for (; it_mem != all_mems.end(); it_mem++) {
    if (it_mem->kind() == Memory::SYSTEM_MEM) {
      valid_mems.push_back(*it_mem);
      
      // select valid processors
      std::vector<Processor> valid_options;
      std::set<Processor> options;
      machine.get_shared_processors(*it_mem, options);
      std::set<Processor>::const_iterator it_proc = options.begin();
      for (; it_proc != options.end(); it_proc++) {
	if (it_proc->kind() == Processor::LOC_PROC)
	  valid_options.push_back(*it_proc);
      }
      MemProc[*it_mem] = valid_options;
    }    
  }
  assert( ! valid_mems.empty() );

  std::cout << "There are " << valid_mems.size()
	    << " machines" << std::endl;
  for (size_t i=0; i<valid_mems.size(); i++) {
    std::cout << "Machine " << i << " has "
	      << MemProc[ valid_mems[i] ].size()
	      << " cores." << std::endl;
  }
}

void SolverMapper::select_task_options(Task *task) {

  //std::cout << "inside select_task()" << std::endl;
  
  task->inline_task   = false;
  task->spawn_task    = false;
  task->map_locally   = false; // turn on remote mapping
  task->profile_task  = false;
  task->task_priority = 0;

  // pick the target memory idexed by task->tag
  // note launch node tasks have negative tags
  assert(valid_mems.size() == 1);
  std::vector<Processor> procs = MemProc[ valid_mems[0] ];
  // select valid processors
  if ( !procs.empty() ) {
    task->target_proc = procs[0];
    task->additional_procs.insert(procs.begin(),
				  procs.end());
  } else {
    // no valid processor available
    assert(false);
  }
}

void SolverMapper::slice_domain(const Task *task, const Domain &domain,
				std::vector<DomainSplit> &slices) {

  //std::cout << "inside slice_domain()" << std::endl;
    
  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);
  std::vector<Processor> split_set;
  for (unsigned idx = 0; idx < 2; idx++)
  {
    split_set.push_back(DefaultMapper::select_random_processor(
                        all_procs, Processor::LOC_PROC, machine));
  }

  DefaultMapper::decompose_index_space(domain, split_set, 
                                        1/*splitting factor*/, slices);

  //std::cout << "slice number: " << slices.size() << std::endl;
  for (std::vector<DomainSplit>::iterator it = slices.begin();
        it != slices.end(); it++)
  {
    Rect<1> rect = it->domain.get_rect<1>();
    if (rect.volume() == 1)
      it->recurse = false;
    else
      it->recurse = true;
  }
}

bool SolverMapper::map_task(Task *task) {

  //std::cout << "inside map_task()" << std::endl;
    
  // Put everything in the system memory
  Memory sys_mem = machine_interface.find_memory_kind
    (task->target_proc, Memory::SYSTEM_MEM);
  
  assert(sys_mem.exists());
  for (unsigned idx = 0; idx < task->regions.size(); idx++) {
    task->regions[idx].target_ranking.push_back(sys_mem);
    task->regions[idx].virtual_map = false;
    task->regions[idx].enable_WAR_optimization = war_enabled;
    task->regions[idx].reduction_list = false;
      		
    // make everything SOA
    task->regions[idx].blocking_factor = 1;
    //task->regions[idx].max_blocking_factor;
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
}
*/
