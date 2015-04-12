#include "init_matrix.hpp"

#include "macros.hpp" // for FIELDID_V
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
  assert(task->arglen == sizeof(TaskArgs));
  assert(task->local_arglen == sizeof(long));

  Point<1> p = task->index_point.get_point<1>();
  printf("point = %d\n", p[0]);

  const long seed = *((const long*)task->local_args);
  printf("random seed = %lu \n", seed);

  
  const TaskArgs blockSize = *((const TaskArgs*)task->args);
  int rows = blockSize.rows;
  int cols = blockSize.cols;
  //printf("block row size = %i\n", rows);
  //printf("block col size = %i\n", cols);
 
  Rect<2> bounds, subrect;
  bounds.lo.x[0] = p[0] * rows;
  bounds.hi.x[0] = (p[0] + 1) * rows - 1;
  bounds.lo.x[1] = 0;
  bounds.hi.x[1] = cols - 1;
  ByteOffset offsets[2];
  double *base = regions[0].get_field_accessor(FIELDID_V).template typeify<double>().template raw_rect_ptr<2>(bounds, subrect, offsets);
  assert(subrect == bounds);
  //printf("ptr = %p (%d, %d)\n", base, offsets[0].offset, offsets[1].offset);

  struct drand48_data buffer;
  assert( srand48_r( seed, &buffer ) == 0 );
  for(int ri = 0; ri < rows; ri++)
    for(int ci = 0; ci < cols; ci++) {
      double *value = base + ri * offsets[0] + ci * offsets[1];
      assert( drand48_r(&buffer, value) == 0 );
    }
}



