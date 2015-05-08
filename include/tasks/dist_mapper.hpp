#ifndef _dist_mapper_hpp
#define _dist_mapper_hpp

#include "default_mapper.h"

using namespace LegionRuntime::HighLevel;

class DistMapper : public DefaultMapper {
public:
  DistMapper(Machine machine, HighLevelRuntime *rt,
	     Processor local, int radix_=2);
public:
  virtual void select_task_options(Task *task);
  virtual void slice_domain(const Task *task, const Domain &domain,
  			    std::vector<DomainSplit> &slices);
  virtual bool map_task(Task *task); 
  //virtual void notify_mapping_result(const Mappable *mappable);
  virtual void notify_mapping_failed(const Mappable *mappable);

private:
  int radix;
  int num_mems;
  std::vector<Memory> valid_mems;
  std::map<Memory, std::vector<Processor> > mem_procs;
};

#endif
