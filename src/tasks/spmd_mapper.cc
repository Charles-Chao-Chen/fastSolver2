#include "legion.h"
#include "utilities.h"
#include "spmd_mapper.hpp"

//Realm::Logger log_solver_mapper("solver_mapper");

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
  // Task options:
  task->inline_task   = false;
  task->spawn_task    = false;
  task->map_locally   = true;
  task->profile_task  = false;
}

#if 0
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
}
#endif


bool SPMDsolverMapper::map_task(Task *task) {

#if 0
  std::cout << "Inside map_task() ..." << std::endl;
  std::cout << "orign: " << task->orig_proc.id
	    << ", current: " << task->current_proc.id
	    << ", target: " << task->target_proc.id
	    << std::endl;
#endif

  //log_solver_mapper.print("map task %s: %i", task->variants->name, task->index_point.point_data[0]);

  Memory sysmem = proc_sysmems[task->target_proc];
  assert(sysmem.exists());
  std::vector<Processor> local_procs = sysmem_local_procs[sysmem];
  task->additional_procs.insert(local_procs.begin(), local_procs.end());
  
  std::vector<RegionRequirement> &regions = task->regions;
  for (std::vector<RegionRequirement>::iterator it = regions.begin();
        it != regions.end(); it++) {
    RegionRequirement &req = *it;

    // Region options:
    req.virtual_map = false;
    req.enable_WAR_optimization = false;
    req.reduction_list = false;

    // Place all regions in local system memory.
    req.target_ranking.push_back(sysmem);
    std::set<FieldID> fields;
    get_field_space_fields(req.parent.get_field_space(), fields);
    req.additional_fields.insert(fields.begin(), fields.end());
  }
  return false;
}


bool SPMDsolverMapper::map_must_epoch(const std::vector<Task*> &tasks,
				      const std::vector<MappingConstraint> &constraints,
				      MappingTagID tag) {

  unsigned tasks_per_sysmem = (tasks.size() + sysmems_list.size() - 1) / sysmems_list.size();
  for (unsigned i = 0; i < tasks.size(); ++i)
  {
    Task* task = tasks[i];
    unsigned index = task->index_point.point_data[0];
    assert(index / tasks_per_sysmem < sysmems_list.size());
    Memory sysmem = sysmems_list[index / tasks_per_sysmem];
    unsigned subindex = index % tasks_per_sysmem;
    assert(subindex < sysmem_local_procs[sysmem].size());
    task->target_proc = sysmem_local_procs[sysmem][subindex];
    map_task(task);
    task->additional_procs.clear();
  }

  typedef std::map<LogicalRegion, Memory> Mapping;
  Mapping mappings;
  for (unsigned i = 0; i < constraints.size(); ++i)
  {
    const MappingConstraint& c = constraints[i];
    if (c.t1->regions[c.idx1].flags & NO_ACCESS_FLAG &&
        c.t2->regions[c.idx2].flags & NO_ACCESS_FLAG)
      continue;

    Memory regmem;
    if (c.t2->regions[c.idx2].flags & NO_ACCESS_FLAG)
      regmem = proc_sysmems[c.t1->target_proc]; // proc_regmems[c.t1->target_proc];
    else if (c.t1->regions[c.idx1].flags & NO_ACCESS_FLAG)
      regmem = proc_sysmems[c.t2->target_proc]; // proc_regmems[c.t2->target_proc];
    else
      assert(0);
    c.t1->regions[c.idx1].target_ranking.clear();
    c.t1->regions[c.idx1].target_ranking.push_back(regmem);
    c.t2->regions[c.idx2].target_ranking.clear();
    c.t2->regions[c.idx2].target_ranking.push_back(regmem);
    mappings[c.t1->regions[c.idx1].region] = regmem;
  }

  for (unsigned i = 0; i < constraints.size(); ++i)
  {
    const MappingConstraint& c = constraints[i];
    if (c.t1->regions[c.idx1].flags & NO_ACCESS_FLAG &&
        c.t2->regions[c.idx2].flags & NO_ACCESS_FLAG)
    {
      Mapping::iterator it =
        mappings.find(c.t1->regions[c.idx1].region);
      assert(it != mappings.end());
      Memory regmem = it->second;
      c.t1->regions[c.idx1].target_ranking.clear();
      c.t1->regions[c.idx1].target_ranking.push_back(regmem);
      c.t2->regions[c.idx2].target_ranking.clear();
      c.t2->regions[c.idx2].target_ranking.push_back(regmem);
    }
  }
  return false;
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
