#ifndef _spmd_mapper_hpp
#define _spmd_mapper_hpp

#include "default_mapper.h"
#include "shim_mapper.h"

using namespace Legion;

class SPMDsolverMapper : public ShimMapper {
public:
  SPMDsolverMapper(Machine machine, Runtime *rt, Processor local,
		   std::vector<Processor>* procs_list,
		   std::vector<Memory>* sysmems_list,
		   std::map<Memory, std::vector<Processor> >* sysmem_local_procs,
		   std::map<Processor, Memory>* proc_sysmems,
		   std::map<Processor, Memory>* proc_regmems);
  virtual void select_task_options(Task *task);
  virtual void slice_domain(const Task *task, const Domain &domain,
			    std::vector<DomainSplit> &slices);
  virtual bool map_task(Task *task);
  virtual bool map_must_epoch(const std::vector<Task*> &tasks,
                              const std::vector<MappingConstraint> &constraints,
                              MappingTagID tag);
  virtual void notify_mapping_failed(const Mappable *mappable);
  //virtual void notify_mapping_result(const Mappable *mappable);
  
private:
  std::vector<Processor> procs_list;
  std::vector<Memory> sysmems_list;
  std::map<Memory, std::vector<Processor> > sysmem_local_procs;
  std::map<Processor, Memory> proc_sysmems;
  std::map<Processor, Memory> proc_regmems;
};

#endif
