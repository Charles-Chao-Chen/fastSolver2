#include "custom_mapper.h"

void register_custom_mapper() {
  HighLevelRuntime::set_registration_callback(mapper_registration);
}
void mapper_registration(Machine machine, HighLevelRuntime *rt,
			 const std::set<Processor> &local_procs)
{
  std::set<Processor>::const_iterator it = local_procs.begin();
  for (; it != local_procs.end(); it++) {
    rt->replace_default_mapper(
	new SolverMapper(machine, rt, *it), *it);
  }
}

SolverMapper::SolverMapper
(Machine m, HighLevelRuntime *rt, Processor p)
  : DefaultMapper(m, rt, p)
{
  typedef std::set<Memory>::const_iterator SMCI;

  std::set<Memory> all_mems;
  machine.get_all_memories(all_mems);
  for (SMCI it = all_mems.begin(); it != all_mems.end(); it++) {
    Memory::Kind kind = it->kind();
    if (kind == Memory::SYSTEM_MEM) {
      valid_mems.push_back(*it);
    }
  }
  assert( ! valid_mems.empty() );
}

void SolverMapper::select_task_options(Task *task)
{
  task->inline_task   = false;
  task->spawn_task    = false;
  task->map_locally   = false; // turn on remote mapping
  task->profile_task  = false;
  task->task_priority = 0;

  // pick the target memory idexed by task->tag
  // note launch node tasks have negative tags
  unsigned taskTag = abs(task->tag);
  assert(taskTag < valid_mems.size());
  Memory mem = valid_mems[taskTag];
  assert(mem != Memory::NO_MEMORY);
  
  // select valid processors
  // TODO: put this into the constructor using std::map
  typedef std::set<Processor>::const_iterator SPCI;
  std::vector<Processor> valid_options;
  std::set<Processor> options;
  machine.get_shared_processors(mem, options);
  for (SPCI it = options.begin(); it != options.end(); ) {
    Processor::Kind kind = it->kind();
    if (kind == Processor::LOC_PROC)
      valid_options.push_back(*it);
    it++;
  }
  
  if ( !valid_options.empty() ) {
    task->target_proc = valid_options[0];
    task->additional_procs.insert(valid_options.begin(),
				  valid_options.end());
  } else {
    // no valid processor available
    task->target_proc = Processor::NO_PROC;
    assert(false);
  }
}

bool SolverMapper::map_task(Task *task)
{    
  // Put everything in the system memory
  Memory sys_mem = 
    machine_interface.find_memory_kind(task->target_proc,
				       Memory::SYSTEM_MEM);
  assert(sys_mem.exists());
  for (unsigned idx = 0; idx < task->regions.size(); idx++)
    {
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
