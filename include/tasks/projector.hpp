#ifndef _projector_hpp
#define _projector_hpp

#include "legion.h"
using namespace LegionRuntime::HighLevel;

static ProjectionID CONTRACTION = 1988;

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

#endif
