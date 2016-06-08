//#include "utilities.h"

#include "new_mapper.hpp"

Realm::Logger log_solver_mapper("solver_mapper");

SolverMapper::SolverMapper(Machine machine, Runtime *rt, Processor local,
			   std::vector<Processor>* _procs_list,
			   std::vector<Memory>* _sysmems_list,
			   std::map<Memory, std::vector<Processor> >* _sysmem_local_procs,
			   std::map<Processor, Memory>* _proc_sysmems,
			   std::map<Processor, Memory>* _proc_regmems)
  : DefaultMapper(rt->get_mapper_runtime(), machine, local),
    procs_list(*_procs_list),
    sysmems_list(*_sysmems_list),
    sysmem_local_procs(*_sysmem_local_procs),
    proc_sysmems(*_proc_sysmems),
  proc_regmems(*_proc_regmems)
{}


void SolverMapper::map_must_epoch(const MapperContext           ctx,
				  const MapMustEpochInput&      input,
				  MapMustEpochOutput&     output)
{
  log_solver_mapper.spew("Default map_must_epoch in %s", get_mapper_name());
  // Figure out how to assign tasks to CPUs first. We know we can't
  // do must epochs for anthing but CPUs at the moment.
  std::map<const Task*,Processor> proc_map;
  if (total_nodes > 1)
    {
      unsigned tasks_per_sysmem = (input.tasks.size() + sysmems_list.size() - 1) / sysmems_list.size();
      for (unsigned i = 0; i < input.tasks.size(); ++i)
	{

	  const Task* task = input.tasks[i];
	  unsigned index = task->index_point.point_data[0];
	  assert(index / tasks_per_sysmem < sysmems_list.size());
	  Memory sysmem = sysmems_list[index / tasks_per_sysmem];
	  unsigned subindex = index % tasks_per_sysmem;
	  assert(subindex < sysmem_local_procs[sysmem].size());
	  output.task_processors[i] = sysmem_local_procs[sysmem][subindex];
          proc_map[input.tasks[i]] = sysmem_local_procs[sysmem][subindex]; 
	  //map_task(task);
	  //task->additional_procs.clear();
	}
    }

  else
    {
      if (input.tasks.size() > local_cpus.size())
        {
          log_solver_mapper.error("Default mapper error. Not enough CPUs for must "
				  "epoch launch of task %s with %ld tasks", 
				  input.tasks[0]->get_task_name(),
				  input.tasks.size());
          assert(false);
        }
      for (unsigned idx = 0; idx < input.tasks.size(); idx++)
        {
          output.task_processors[idx] = local_cpus[idx];
          proc_map[input.tasks[idx]] = local_cpus[idx];
        }
    }
  // Now let's map the constraints, find one requirement to use for
  // mapping each of the constraints, but get the set of fields we
  // care about and the set of logical regions for all the requirements
  for (unsigned cid = 0; cid < input.constraints.size(); cid++)
    {
      const MappingConstraint &constraint = input.constraints[cid];
      std::vector<Legion::Mapping::PhysicalInstance> &constraint_mapping = 
	output.constraint_mappings[cid];
      // Figure out which task and region requirement to use as the 
      // basis for doing the mapping
      Task *base_task = NULL;
      unsigned base_index = 0;
      Processor base_proc = Processor::NO_PROC;
      std::set<LogicalRegion> needed_regions;
      std::set<FieldID> needed_fields;
      for (unsigned idx = 0; idx < constraint.constrained_tasks.size(); idx++)
        {
          Task *task = constraint.constrained_tasks[idx];
          unsigned req_idx = constraint.requirement_indexes[idx];
          if ((base_task == NULL) && (!task->regions[req_idx].is_no_access()))
	    {
	      base_task = task;
	      base_index = req_idx;
	      base_proc = proc_map[task];
	    }
          needed_regions.insert(task->regions[req_idx].region);
          needed_fields.insert(task->regions[req_idx].privilege_fields.begin(),
                               task->regions[req_idx].privilege_fields.end());
        }
      // If there wasn't a region requirement that wasn't no access just 
      // pick the first one since this case doesn't make much sense anyway
      if (base_task == NULL)
        {
          base_task = constraint.constrained_tasks[0];
          base_index = constraint.requirement_indexes[0];
          base_proc = proc_map[base_task];
        }
      Memory target_memory = default_policy_select_target_memory(ctx, 
								 base_proc);
      VariantInfo info = default_find_preferred_variant(*base_task, ctx, 
							true/*needs tight bound*/, true/*cache*/, Processor::LOC_PROC);
      const TaskLayoutConstraintSet &layout_constraints = 
	runtime->find_task_layout_constraints(ctx, base_task->task_id, 
					      info.variant);
      if (needed_regions.size() == 1)
        {
          // If there was just one region we can use the base region requirement
          if (!default_create_custom_instances(ctx, base_proc, target_memory,
					       base_task->regions[base_index], base_index, needed_fields,
					       layout_constraints, true/*needs check*/, constraint_mapping))
	    {
	      log_solver_mapper.error("Default mapper error. Unable to make instance(s) "
				      "in memory " IDFMT " for index %d of constrained "
				      "task %s (ID %lld) in must epoch launch.",
				      target_memory.id, base_index,
				      base_task->get_task_name(), 
				      base_task->get_unique_id());
	      assert(false);
	    }
        }
      else
        {
          // Otherwise we need to find a common region that will satisfy all
          // the needed regions
          RegionRequirement copy_req = base_task->regions[base_index];
          copy_req.region = default_find_common_ancestor(ctx, needed_regions);
          if (!default_create_custom_instances(ctx, base_proc, target_memory,
					       copy_req, base_index, needed_fields, layout_constraints,
					       true/*needs check*/, constraint_mapping))
	    {
	      log_solver_mapper.error("Default mapper error. Unable to make instance(s) "
				      "in memory " IDFMT " for index %d of constrained "
				      "task %s (ID %lld) in must epoch launch.",
				      target_memory.id, base_index,
				      base_task->get_task_name(), 
				      base_task->get_unique_id());
	      assert(false);
	    }
        }
    }
}

// suggested by Elliott because of potential runtime bugs
bool SolverMapper::default_policy_select_close_virtual(const MapperContext,
                                                       const Close &)
{
  return false;  
}

Processor SolverMapper::default_policy_select_initial_processor
(MapperContext ctx, const Task &task) {
  
  VariantInfo info = 
    default_find_preferred_variant(task, ctx, false/*needs tight*/);
  // If we are the right kind then we return ourselves
  if (info.proc_kind == local_kind)
    return local_proc;
  // Otherwise pick a local one of the right type
  switch (info.proc_kind)
    {
    case Processor::LOC_PROC:
      {
	assert(!local_cpus.empty());
	return default_select_random_processor(local_cpus); 
      }
    case Processor::TOC_PROC:
      {
	assert(!local_gpus.empty());
	return default_select_random_processor(local_gpus);
      }
    case Processor::IO_PROC:
      {
	assert(!local_ios.empty());
	return default_select_random_processor(local_ios);
      }
    default:
      assert(false);
    }
  return Processor::NO_PROC;
}
