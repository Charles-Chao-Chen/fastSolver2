#include "projector.hpp"

void registration_callback(Machine machine, HighLevelRuntime *runtime,
                           const std::set<Processor> &local_procs) {

  runtime->register_projection_functor(
    PROJECTION_REDUCE,
    new ProjectionReduce(runtime));
}

