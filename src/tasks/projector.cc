#include "projector.hpp"

const ProjectionID CONTRACTION = 1988;

void register_projector(Machine machine, HighLevelRuntime *runtime,
			const std::set<Processor> &local_procs) {

  runtime->register_projection_functor(
    CONTRACTION, new Contraction(runtime));
}

Contraction::Contraction(HighLevelRuntime *runtime)
  : ProjectionFunctor(runtime) {}

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
  int clrSize = *((int*)task->args);
  int color = point.point_data[0] / clrSize;

#ifdef DEBUG
  printf("domain size: %d\tpoint: %d\tblock: %d\n",
	 size, point.point_data[0], color);
#endif
  
  return runtime->get_logical_subregion_by_color(ctx, partition, color);
}
