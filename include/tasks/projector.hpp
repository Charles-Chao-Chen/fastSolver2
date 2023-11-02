#ifndef _projector_hpp
#define _projector_hpp

#include "legion.h"
using namespace Legion;

extern const ProjectionID CONTRACTION;

class Contraction : public ProjectionFunctor {
public:
  
  Contraction(Runtime *runtime);

  virtual LogicalRegion project(Context ctx, Task *task,
                                unsigned index,
                                LogicalRegion upper_bound,
                                const DomainPoint &point);

  virtual LogicalRegion project(Context ctx, Task *task,
                                unsigned index,
                                LogicalPartition upper_bound,
                                const DomainPoint &point);

  unsigned get_depth() const;
};

#endif
