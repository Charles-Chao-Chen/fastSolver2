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

  void slice_task(const MapperContext      ctx,
		  const Task&              task, 
		  const SliceTaskInput&    input,
		  SliceTaskOutput&   output);
  
  void default_slice_task(const Task &task,
			  const std::vector<Processor> &local,
			  const std::vector<Processor> &remote,
			  const SliceTaskInput& input,
			  SliceTaskOutput &output,
			  std::map<Domain,std::vector<TaskSlice> >
			  &cached_slices) const;
	
private:
  std::vector<Processor> procs_list;
  std::vector<Memory> sysmems_list;
  std::map<Memory, std::vector<Processor> > sysmem_local_procs;
  std::map<Processor, Memory> proc_sysmems;
  std::map<Processor, Memory> proc_regmems;
};


#endif
