#ifndef _new_mapper_hpp
#define _new_mapper_hpp

#include "default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

class SolverMapper : public DefaultMapper {
public:
  SolverMapper(Machine machine, Runtime *rt, Processor local,
	       std::vector<Processor>* procs_list,
	       std::vector<Memory>* sysmems_list,
	       std::map<Memory, std::vector<Processor> >* sysmem_local_procs,
	       std::map<Processor, Memory>* proc_sysmems,
	       std::map<Processor, Memory>* proc_regmems);

#if 1
  virtual void map_must_epoch(const MapperContext           ctx,
			      const MapMustEpochInput&      input,
			      MapMustEpochOutput&     output);
#endif

public:
  virtual bool default_policy_select_close_virtual(const MapperContext,
						   const Close &);
  virtual Processor default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task);

private:
  std::vector<Processor> procs_list;
  std::vector<Memory> sysmems_list;
  std::map<Memory, std::vector<Processor> > sysmem_local_procs;
  std::map<Processor, Memory> proc_sysmems;
  std::map<Processor, Memory> proc_regmems;
};


#endif
