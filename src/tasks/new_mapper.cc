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
	output.task_processors[idx] = local_cpus[idx];
    }

  // Now let's map all the constraints first, and then we'll call map
  // task for all the tasks and tell it that we already premapped the
  // constrainted instances
  for (unsigned cid = 0; cid < input.constraints.size(); cid++)
    {
      const MappingConstraint &constraint = input.constraints[cid];
      std::vector<Legion::Mapping::PhysicalInstance> &constraint_mapping =
	output.constraint_mappings[cid];
      int index1 = -1, index2 = -1;
      for (unsigned idx = 0; (idx < input.tasks.size()) &&
	     ((index1 == -1) || (index2 == -1)); idx++)
        {
          if (constraint.t1 == input.tasks[idx])
            index1 = idx;
          if (constraint.t2 == input.tasks[idx])
            index2 = idx;
        }
      assert((index1 >= 0) && (index2 >= 0));
      // Figure out which memory to use
      // TODO: figure out how to use registered memory in the multi-node case
      Memory target1 = default_policy_select_target_memory(ctx,
							   output.task_processors[index1]);
      Memory target2 = default_policy_select_target_memory(ctx,
							   output.task_processors[index2]);
      // Pick our target memory
      Memory target_memory = Memory::NO_MEMORY;
      if (target1 != target2)
        {
          // See if one of them is not no access so we can pick the other
          if (constraint.t1->regions[constraint.idx1].is_no_access())
            target_memory = target2;
          else if (constraint.t2->regions[constraint.idx2].is_no_access())
            target_memory = target1;
          else
	    {
	      log_solver_mapper.error("Default mapper error. Unable to pick a common "
			       "memory for tasks %s (ID %lld) and %s (ID %lld) "
			       "in a must epoch launch. This will require a "
			       "custom mapper.", constraint.t1->get_task_name(),
			       constraint.t1->get_unique_id(), 
			       constraint.t2->get_task_name(), 
			       constraint.t2->get_unique_id());
	      assert(false);
	    }
        }
      else // both the same so this is easy
	target_memory = target1;
      assert(target_memory.exists());
      // Figure out the variants that are going to be used by the two tasks    
      VariantInfo info1 = find_preferred_variant(*constraint.t1, ctx,
						 true/*needs tight bound*/, Processor::LOC_PROC);
      VariantInfo info2 = find_preferred_variant(*constraint.t2, ctx,
						 true/*needs tight_bound*/, Processor::LOC_PROC);
      // Map it the one way and filter the other so that we can make sure
      // that they are both going to use the same instance
      std::set<FieldID> needed_fields = 
	constraint.t1->regions[constraint.idx1].privilege_fields;
      needed_fields.insert(
			   constraint.t2->regions[constraint.idx2].privilege_fields.begin(),
			   constraint.t2->regions[constraint.idx2].privilege_fields.end());
      const TaskLayoutConstraintSet &layout_constraints1 = 
	runtime->find_task_layout_constraints(ctx, 
						     constraint.t1->task_id, info1.variant);
      if (!default_create_custom_instances(ctx, 
					   output.task_processors[index1], target_memory,
					   constraint.t1->regions[constraint.idx1], constraint.idx1,
					   needed_fields, layout_constraints1, true/*needs check*/,
					   constraint_mapping))
        {
          log_solver_mapper.error("Default mapper error. Unable to make instance(s) "
                           "in memory " IDFMT " for index %d of constrained "
                           "task %s (ID %lld) in must epoch launch.",
                           target_memory.id, constraint.idx1, 
                           constraint.t1->get_task_name(),
                           constraint.t1->get_unique_id());
          assert(false);
        }
      // Copy the results over and make sure they are still good 
      const size_t num_instances = constraint_mapping.size();
      assert(num_instances > 0);
      std::set<FieldID> missing_fields;
      runtime->filter_instances(ctx, *constraint.t2, constraint.idx2,
				       info2.variant, constraint_mapping, missing_fields);
      if (num_instances != constraint_mapping.size())
        {
          log_solver_mapper.error("Default mapper error. Unable to make instance(s) "
                           "for index %d of constrained task %s (ID %lld) in "
                           "must epoch launch. Most likely this is because "
                           "conflicting constraints are requested for regions "
                           "which must be mapped to the same instance. You "
                           "will need to write a custom mapper to fix this.",
                           constraint.idx2, constraint.t2->get_task_name(),
                           constraint.t2->get_unique_id());
          assert(false);
        }
    }
}

// suggested by Elliott because of potential runtime bugs
bool SolverMapper::default_policy_select_close_virtual(const MapperContext,
                                                       const Close &)
{
  return false;  
}

