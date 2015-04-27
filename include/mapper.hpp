#include "default_mapper.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;


class SolverMapper : public DefaultMapper {
public:
  SolverMapper(Machine machine, 
      HighLevelRuntime *rt, Processor local);
public:
  virtual void select_task_options(Task *task);
  virtual void slice_domain(const Task *task, const Domain &domain,
			    std::vector<DomainSplit> &slices);
  virtual bool map_task(Task *task); 
  //virtual void notify_mapping_result(const Mappable *mappable);
  virtual void notify_mapping_failed(const Mappable *mappable);
 private:
  std::vector<Memory> valid_mems;
};


void register_mapper();
void mapper_registration(Machine machine, HighLevelRuntime *rt,
			 const std::set<Processor> &local_procs);


