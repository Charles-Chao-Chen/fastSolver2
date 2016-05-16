#include "solver_tasks.hpp"


void create_spmd_solver_mapper(Machine machine, HighLevelRuntime *rt,
			       const std::set<Processor> &local_procs) {

  create_projector(machine, rt, local_procs);
  
  std::vector<Processor>* procs_list = new std::vector<Processor>();
  std::vector<Memory>* sysmems_list = new std::vector<Memory>();
  std::map<Memory, std::vector<Processor> >* sysmem_local_procs =
    new std::map<Memory, std::vector<Processor> >();
  std::map<Processor, Memory>* proc_sysmems = new std::map<Processor, Memory>();
  std::map<Processor, Memory>* proc_regmems = new std::map<Processor, Memory>();


  std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
  machine.get_proc_mem_affinity(proc_mem_affinities);

  for (unsigned idx = 0; idx < proc_mem_affinities.size(); ++idx) {
    Machine::ProcessorMemoryAffinity& affinity = proc_mem_affinities[idx];
    if (affinity.p.kind() == Processor::LOC_PROC) {
      if (affinity.m.kind() == Memory::SYSTEM_MEM) {
        (*proc_sysmems)[affinity.p] = affinity.m;
        if (proc_regmems->find(affinity.p) == proc_regmems->end())
          (*proc_regmems)[affinity.p] = affinity.m;
      }
      else if (affinity.m.kind() == Memory::REGDMA_MEM)
        (*proc_regmems)[affinity.p] = affinity.m;
    }
  }

  for (std::map<Processor, Memory>::iterator it = proc_sysmems->begin();
       it != proc_sysmems->end(); ++it) {
    procs_list->push_back(it->first);
    (*sysmem_local_procs)[it->second].push_back(it->first);
  }

  for (std::map<Memory, std::vector<Processor> >::iterator it =
        sysmem_local_procs->begin(); it != sysmem_local_procs->end(); ++it)
    sysmems_list->push_back(it->first);

  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    SPMDsolverMapper* mapper = new SPMDsolverMapper(machine, rt, *it,
                                              procs_list,
                                              sysmems_list,
                                              sysmem_local_procs,
                                              proc_sysmems,
                                              proc_regmems);
    rt->replace_default_mapper(mapper, *it);
  }
}

void create_mapper(Machine machine, HighLevelRuntime *rt,
			   const std::set<Processor> &local_procs) {

  std::set<Processor>::const_iterator it = local_procs.begin();
  for (; it != local_procs.end(); it++) {
    SolverMapper *mapper = new SolverMapper(machine, rt, *it);
    rt->replace_default_mapper(mapper,*it);
  }
}

void create_projector(Machine machine, HighLevelRuntime *rt,
			   const std::set<Processor> &local_procs) {    
  rt->register_projection_functor
    (CONTRACTION, new Contraction(rt));
}

void register_solver_tasks() {
  InitMatrixTask::register_tasks();
  DenseBlockTask::register_tasks();
  AddMatrixTask::register_tasks();
  ClearMatrixTask::register_tasks();
  ScaleMatrixTask::register_tasks();
  DisplayMatrixTask::register_tasks();
  
  LeafSolveTask::register_tasks();
  NodeSolveTask::register_tasks();
  NodeSolveRegionTask::register_tasks();
  GemmTask::register_tasks();
  GemmInplaceTask::register_tasks();
  GemmRedTask::register_tasks();
  GemmBroTask::register_tasks();
  Add::register_operator();
  HighLevelRuntime::set_registration_callback(create_spmd_solver_mapper);
}
