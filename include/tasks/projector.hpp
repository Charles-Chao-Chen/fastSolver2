#ifndef _projector_hpp
#define _projector_hpp

#include "legion.h"
using namespace LegionRuntime::HighLevel;

extern const ProjectionID CONTRACTION;

void register_projector(Machine machine, HighLevelRuntime *runtime,
			const std::set<Processor> &local_procs);

class Contraction : public ProjectionFunctor {
public:
  
  Contraction(HighLevelRuntime *runtime);

  virtual LogicalRegion project(Context ctx, Task *task,
                                unsigned index,
                                LogicalRegion upper_bound,
                                const DomainPoint &point);

  virtual LogicalRegion project(Context ctx, Task *task,
                                unsigned index,
                                LogicalPartition upper_bound,
                                const DomainPoint &point);
};
/*
extern const ProjectionID EXPAND;

void register_expand(Machine machine, HighLevelRuntime *runtime,
		     const std::set<Processor> &local_procs);

class Expand : public ProjectionFunctor {
public:
  
  Expand(HighLevelRuntime *runtime);

  virtual LogicalRegion project(Context ctx, Task *task,
                                unsigned index,
                                LogicalRegion upper_bound,
                                const DomainPoint &point);

  virtual LogicalRegion project(Context ctx, Task *task,
                                unsigned index,
                                LogicalPartition upper_bound,
                                const DomainPoint &point);
};
*/
#endif
