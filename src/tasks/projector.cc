#include "projector.hpp"

const ProjectionID CONTRACTION = 1988;

Contraction::Contraction(HighLevelRuntime *runtime)
  : ProjectionFunctor(runtime) {
  //std::cout<<"Register projection functor with ID: "<<CONTRACTION<<std::endl;
}

LogicalRegion Contraction::project(Context ctx, Task *task,
				   unsigned index,
				   LogicalRegion upper_bound,
				   const DomainPoint &point) {
  assert(false && "unimplemented");
}

LogicalRegion Contraction::project(Context ctx, Task *task,
				   unsigned index,
				   LogicalPartition partition,
				   const DomainPoint &point) {

  // pass in the size of the launch domain
  int *args = (int*)task->args;
  int clrSize = *args;
  int plevel  = *(++args);
  //printf("colorSize: %d, partition level: %d\n", clrSize, plevel);
  
  int color = point.point_data[0] / clrSize;
  if (plevel == 1) {
    return runtime->get_logical_subregion_by_color(ctx, partition, color);
  }
  if (plevel == 2) {
    int clr1 = color / 2;
    LogicalRegion lr1 = runtime->get_logical_subregion_by_color(ctx, partition, clr1);
    LogicalPartition lp = runtime->get_logical_partition_by_color(ctx, lr1, 0);
    return runtime->get_logical_subregion_by_color(ctx, lp, color%2);
  }
  assert(false);
}
