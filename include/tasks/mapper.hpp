#ifndef _mapper_hpp
#define _mapper_hpp

#include "default_mapper.h"

using namespace Legion;

class SolverMapper : public DefaultMapper {
public:
  SolverMapper(Machine machine, 
	       Runtime *rt, Processor local);
public:
  virtual void select_task_options(Task *task);
  virtual void slice_domain(const Task *task, const Domain &domain,
  			    std::vector<DomainSplit> &slices);
  virtual bool map_task(Task *task); 
  //virtual void notify_mapping_result(const Mappable *mappable);
  virtual void notify_mapping_failed(const Mappable *mappable);

private:
  int num_mems;
  std::vector<Memory> valid_mems;
  std::map<Memory, std::vector<Processor> > mem_procs;
};

#endif
