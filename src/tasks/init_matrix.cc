#include "init_matrix.hpp"

#include <assert.h>

int InitMatrixTask::TASKID;

InitMatrixTask::InitMatrixTask(Domain domain,
			       TaskArgument global_arg,
			       ArgumentMap arg_map,
			       Predicate pred,
			       bool must,
			       MapperID id,
			       MappingTagID tag)
  
  : IndexLauncher(TASKID, domain, global_arg,
		  arg_map, pred, must, id, tag) {}

void InitMatrixTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <InitMatrixTask::cpu_task>(AUTO_GENERATE_ID,
			       Processor::LOC_PROC, 
			       false,
			       true,
			       AUTO_GENERATE_ID,
			       TaskConfigOptions(true/*leaf*/),
			       "Init_Matrix");

  //#ifndef SHOW_REGISTER_TASKS
  printf("Register task %d : Init_Matrix\n", TASKID);
  //#endif
}

void InitMatrixTask::cpu_task(const Task *task,
			      const std::vector<PhysicalRegion> &regions,
			      Context ctx, HighLevelRuntime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->local_arglen == sizeof(long));

  Point<1> p = task->index_point.get_point<1>();
  printf("point = %d\n", p[0]);

  const long seed = *((const long*)task->local_args);
  printf("random seed = %lu \n", seed);

  /*
  int rb = args->matrix.rblock;
  int cb = args->matrix.cblock;
  
  Rect<2> bounds, subrect;
  bounds.lo.x[0] = p[0] * rb;
  bounds.lo.x[1] = p[1] * cb;
  bounds.hi.x[0] = (p[0] + 1) * rb - 1;
  bounds.hi.x[1] = (p[1] + 1) * cb - 1;
  ByteOffset offsets[2];
  T *base = regions[0].get_field_accessor(FIELDID_V).template typeify<T>().template raw_rect_ptr<2>(bounds, subrect, offsets);
  assert(subrect == bounds);
#ifdef DEBUG_POINTERS
  printf("ptr = %p (%d, %d)\n", base, offsets[0].offset, offsets[1].offset);
#endif

  for(int ri = 0; ri < args->matrix.rblock; ri++)
    for(int ci = 0; ci < args->matrix.cblock; ci++)
      *(base + ri * offsets[0] + ci * offsets[1]) = args->clear_val;
*/
}



