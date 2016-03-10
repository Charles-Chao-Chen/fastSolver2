#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <unistd.h>

#include "reduce_add.hpp"
#include "legion.h"
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;


enum {
  TOP_LEVEL_TASK_ID,
  SPMD_TASK_ID,
  INIT_FIELD_TASK_ID,
  LOCAL_SUM_TASK_ID,
  COMPUTE_NODE_TASK_ID,
  SHIFT_TASK_ID,
  CHECK_FIELD_TASK_ID,
};

enum {
  FID_VAL,
  FID_GHOST,
};

enum {
  GHOST_LEFT,
  GHOST_RIGHT,
};

struct SPMDArgs {
public:
  PhaseBarrier redop_finish;
  PhaseBarrier node_finish;
  int num_elements;
  int num_subregions;
  int num_steps;
};

void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {  

  int num_elements = 1024;
  int num_subregions = 4;
  int num_steps = 10;
  // Check for any command line arguments
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-b"))
        num_subregions = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-s"))
        num_steps = atoi(command_args.argv[++i]);
    }
  }
  assert(num_elements % num_subregions == 0);
  // This algorithm needs at least two sub-regions to work
  assert(num_subregions > 1);
  printf("Running stencil computation for %d elements for %d steps...\n", 
          num_elements, num_steps);
  printf("Partitioning data into %d sub-regions...\n", num_subregions);

  // we're going to use a must epoch launcher, so we need at least as many
  //  processors in our system as we have subregions - check that now
  std::set<Processor> all_procs;
  Realm::Machine::get_machine().get_all_processors(all_procs);
  int num_loc_procs = 0;
  for(std::set<Processor>::const_iterator it = all_procs.begin();
      it != all_procs.end();
      it++)
    if((*it).kind() == Processor::LOC_PROC)
      num_loc_procs++;

  if(num_loc_procs < num_subregions) {
    printf("FATAL ERROR: This test uses a must epoch launcher, which requires\n");
    printf("  a separate Realm processor for each subregion.  %d of the necessary\n",
	   num_loc_procs);
    printf("  %d are available.  Please rerun with '-ll:cpu %d'.\n",
	   num_subregions, num_subregions);
    exit(1);
  }

  // Create region for global reduction/broadcast results
  Rect<1> elem_rect(Point<1>(0),Point<1>(0));
  IndexSpace is = runtime->create_index_space(ctx,
                          Domain::from_rect<1>(elem_rect));
  runtime->attach_name(is, "is");
  
  FieldSpace ghost_fs = runtime->create_field_space(ctx);
  runtime->attach_name(ghost_fs, "ghost_fs");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, ghost_fs);
    allocator.allocate_field(sizeof(double),FID_GHOST);
    runtime->attach_name(ghost_fs, FID_GHOST, "GHOST");
  }

  LogicalRegion ghost_lr = 
     runtime->create_logical_region(ctx, is, ghost_fs);
  
  MustEpochLauncher must_epoch_launcher;
  std::vector<SPMDArgs> args(num_subregions);

  /**************************************/
  std::vector<PhaseBarrier> bar_redop;
  bar_redop.push_back(runtime->create_phase_barrier(ctx, num_subregions));
  
  std::vector<PhaseBarrier> bar_node;
  bar_node.push_back(runtime->create_phase_barrier(ctx, 1));
  
  for (int color=0; color<num_subregions; color++) {

    args[color].num_elements = num_elements;
    args[color].num_subregions = num_subregions;
    args[color].num_steps = num_steps;
    args[color].redop_finish = bar_redop[0];
    args[color].node_finish = bar_node[0];
    
    TaskLauncher spmd_launcher(SPMD_TASK_ID, TaskArgument(&args[color],sizeof(SPMDArgs)));
    spmd_launcher.add_region_requirement(
          RegionRequirement(ghost_lr, READ_WRITE, 
                            SIMULTANEOUS, ghost_lr));
    spmd_launcher.region_requirements[0].flags |= NO_ACCESS_FLAG;
    spmd_launcher.add_index_requirement(IndexSpaceRequirement(is,
							      NO_MEMORY,
							      is));
    spmd_launcher.add_field(0, FID_GHOST);
    DomainPoint point(color);
    must_epoch_launcher.add_single_task(point, spmd_launcher);
  }

  FutureMap fm = runtime->execute_must_epoch(ctx, must_epoch_launcher);
  fm.wait_all_results();
  printf("Test completed.\n");

  runtime->destroy_index_space(ctx, is);
  runtime->destroy_field_space(ctx, ghost_fs);
}

void spmd_detrend(const Task *task,
		  const std::vector<PhysicalRegion> &regions,
		  Context ctx, HighLevelRuntime *runtime) {  

  int point = task->index_point.get_index();
  std::cout<<"Inside spmd_task["<<point<<"]\n";

  runtime->unmap_all_regions(ctx);

  SPMDArgs *args = (SPMDArgs*)task->args;
  int num_elements = args->num_elements;
  int num_subregions = args->num_subregions;
  //int num_steps = args->num_steps;

  LogicalRegion ghost_lr = task->regions[0].region;
  
  // Create the logical region that we'll use for our data
  int local_size = num_elements/num_subregions;
  Rect<1> elem_rect(Point<1>(0),Point<1>(local_size-1));
  IndexSpace local_is = runtime->create_index_space(ctx,
                          Domain::from_rect<1>(elem_rect));
  IndexSpace ghost_is = task->regions[0].region.get_index_space();
  
  char buf[16];
  sprintf(buf, "local_is_%d", point);
  runtime->attach_name(local_is, buf);
  
  FieldSpace local_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, local_fs);
    allocator.allocate_field(sizeof(double),FID_VAL);
    runtime->attach_name(local_fs, FID_VAL, "VAL");
  }

  FieldSpace sum_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, sum_fs);
    allocator.allocate_field(sizeof(double),FID_VAL);
    runtime->attach_name(local_fs, FID_VAL, "VAL");
  }
  
  LogicalRegion local_lr = 
     runtime->create_logical_region(ctx, local_is, local_fs);  
  sprintf(buf, "local_lr_%d", point);
  runtime->attach_name(local_lr, buf);

  LogicalRegion sum_lr = 
     runtime->create_logical_region(ctx, ghost_is, sum_fs);  
  sprintf(buf, "sum_lr_%d", point);
  runtime->attach_name(sum_lr, buf);

  // Launch a task to initialize our field with some data
  TaskLauncher init_launcher(INIT_FIELD_TASK_ID,
			     TaskArgument(&point, sizeof(int)));
  init_launcher.add_region_requirement(
        RegionRequirement(local_lr, WRITE_DISCARD,
                          EXCLUSIVE, local_lr));
  init_launcher.add_field(0, FID_VAL);
  runtime->execute_task(ctx, init_launcher);

  // Launch a task to compute the local sum
  TaskLauncher local_sum_launcher(LOCAL_SUM_TASK_ID,
				  TaskArgument(&local_size, sizeof(int)));
  local_sum_launcher.add_region_requirement(
        RegionRequirement(local_lr, READ_ONLY,
                          EXCLUSIVE, local_lr));
  local_sum_launcher.add_field(0, FID_VAL);
  local_sum_launcher.add_region_requirement(
        RegionRequirement(sum_lr, WRITE_DISCARD,
                          EXCLUSIVE, sum_lr));
  local_sum_launcher.add_field(1, FID_VAL);    
  runtime->execute_task(ctx, local_sum_launcher);

  /**************************************/

  AcquireLauncher acquire_launcher(ghost_lr,
				   ghost_lr,
				   regions[0]);
  acquire_launcher.add_field(FID_GHOST);
  // The acquire operation needs to wait for the data to
  // be ready to consume, so wait on the next phase of the
  // ready barrier.
  /*
  // no need to wait for the first iteration
  args->wait_ready[idx] = 
        runtime->advance_phase_barrier(ctx, args->wait_ready[idx]);
  acquire_launcher.add_wait_barrier(args->wait_ready[idx]);
  */
  //acuqire_launcher.add_arrival_barrier(args->notify_ready[idx]);
  runtime->issue_acquire(ctx, acquire_launcher);
  //args->notify_ready[idx] = 
  //runtime->advance_phase_barrier(ctx, args->notify_ready[idx]);

  /**************************************/
      
  CopyLauncher copy_launcher;
  copy_launcher.add_copy_requirements(
          RegionRequirement(sum_lr, READ_ONLY,
                            EXCLUSIVE, sum_lr),
          RegionRequirement(ghost_lr, REDOP_ADD,
                            EXCLUSIVE, ghost_lr));
  copy_launcher.add_src_field(0, FID_VAL);
  copy_launcher.add_dst_field(0, FID_GHOST);
  // It's not safe to issue the copy until we know
  // that the destination instance is empty. Only
  // need to do this after the first iteration.

  // advance the barrier first - we're waiting for the next phase
  //  to start
  //args->wait_empty[idx] = 
  //      runtime->advance_phase_barrier(ctx, args->wait_empty[idx]);
  //copy_launcher.add_wait_barrier(args->wait_empty[idx]);

  // When we are done with the copy, signal that the
  // destination instance is now ready
  runtime->issue_copy_operation(ctx, copy_launcher);
  // Once we've issued our copy operation, advance both of
  // the barriers to the next generation.

  /**************************************/

  ReleaseLauncher release_launcher(ghost_lr,
				   ghost_lr,
				   regions[0]);
  release_launcher.add_field(FID_GHOST);
  // On all but the last iteration we need to signal that
  // we have now consumed the ghost instances and it is
  // safe to issue the next copy.

  release_launcher.add_arrival_barrier(args->redop_finish);
  runtime->issue_release(ctx, release_launcher);
  args->redop_finish = 
      runtime->advance_phase_barrier(ctx, args->redop_finish);
    
  /**************************************/

  if (point == 0) {
    TaskLauncher node_launcher(COMPUTE_NODE_TASK_ID,
			       TaskArgument(&num_elements, sizeof(int)));
    node_launcher.add_region_requirement(
        RegionRequirement(ghost_lr, READ_WRITE,
			  EXCLUSIVE, ghost_lr));
    node_launcher.add_field(0, FID_GHOST);

    node_launcher.add_wait_barrier(args->redop_finish);
    
    node_launcher.add_arrival_barrier(args->node_finish);
    runtime->execute_task(ctx, node_launcher);
  }

  TaskLauncher shift_launcher(SHIFT_TASK_ID,
			      TaskArgument(NULL, 0));
  shift_launcher.add_region_requirement(
      RegionRequirement(ghost_lr, READ_ONLY,
			EXCLUSIVE, ghost_lr));
  shift_launcher.add_field(0, FID_GHOST);
  shift_launcher.add_region_requirement(
      RegionRequirement(local_lr, READ_WRITE,
			EXCLUSIVE, local_lr));
  shift_launcher.add_field(1, FID_VAL);

  args->node_finish = 
      runtime->advance_phase_barrier(ctx, args->node_finish);
  shift_launcher.add_wait_barrier(args->node_finish);
  runtime->execute_task(ctx, shift_launcher);


  TaskLauncher check_launcher(CHECK_FIELD_TASK_ID,
			      TaskArgument(NULL, 0));
  check_launcher.add_region_requirement(
      RegionRequirement(local_lr, READ_ONLY,
			EXCLUSIVE, local_lr));
  check_launcher.add_field(0, FID_VAL);
  runtime->execute_task(ctx, check_launcher);    
}

void init_field_task(const Task *task,
		     const std::vector<PhysicalRegion> &regions,
		     Context ctx, HighLevelRuntime *runtime) {  

  std::cout<<"Inside init_field_task()"<<std::endl;
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  int point = *((int*)task->args);
  
  RegionAccessor<AccessorType::Generic, double> acc = 
    regions[0].get_field_accessor(fid).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();
  for (GenericPointInRectIterator<1> pir(rect); pir; pir++)
  {    
    double value = (double)(pir.p[0]) + point;
    acc.write(DomainPoint::from_point<1>(pir.p), value);
  }
}

void local_sum_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {  

  assert(regions.size() == 2); 
  assert(task->regions.size() == 2);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 1);
  
  FieldID read_fid = *(task->regions[0].privilege_fields.begin());
  FieldID write_fid = *(task->regions[1].privilege_fields.begin());
  int size = *((int*)task->args);
  
  RegionAccessor<AccessorType::Generic, double> read_acc = 
    regions[0].get_field_accessor(read_fid).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> write_acc = 
    regions[1].get_field_accessor(write_fid).typeify<double>();
  
  Domain read_dom = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  Domain write_dom = runtime->get_index_space_domain(ctx, 
      task->regions[1].region.get_index_space());
  Rect<1> read_rect = read_dom.get_rect<1>();
  Rect<1> write_rect = write_dom.get_rect<1>();
  double sum = 0.0;
  for (GenericPointInRectIterator<1> pir(read_rect); pir; pir++)
  {
    sum += read_acc.read(DomainPoint::from_point<1>(pir.p));
  }
  for (GenericPointInRectIterator<1> pir(write_rect); pir; pir++)
  {
    write_acc.write(DomainPoint::from_point<1>(pir.p), sum);
  }
  std::cout<<"Inside local_sum_task(), local average: "
	   <<sum/size<<std::endl;  
}

void compute_node_task(const Task *task,
		      const std::vector<PhysicalRegion> &regions,
		      Context ctx, HighLevelRuntime *runtime) {  

  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  int *size = (int*)task->args;
  
  RegionAccessor<AccessorType::Generic, double> acc = 
    regions[0].get_field_accessor(fid).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();
  for (GenericPointInRectIterator<1> pir(rect); pir; pir++)
  {
    double value = acc.read(DomainPoint::from_point<1>(pir.p));
    acc.write(DomainPoint::from_point<1>(pir.p), value/(*size));
    std::cout<<"Inside compute_node_task(), global average: "
	     <<value/(*size)<<std::endl;
  }
}

void shift_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime) {  

  assert(regions.size() == 2); 
  assert(task->regions.size() == 2);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 1);
  
  FieldID read_fid = *(task->regions[0].privilege_fields.begin());
  FieldID write_fid = *(task->regions[1].privilege_fields.begin());
  
  RegionAccessor<AccessorType::Generic, double> read_acc = 
    regions[0].get_field_accessor(read_fid).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> write_acc = 
    regions[1].get_field_accessor(write_fid).typeify<double>();
  
  Domain read_dom = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  Domain write_dom = runtime->get_index_space_domain(ctx, 
      task->regions[1].region.get_index_space());
  Rect<1> read_rect = read_dom.get_rect<1>();
  Rect<1> write_rect = write_dom.get_rect<1>();
  double avg;
  for (GenericPointInRectIterator<1> pir(read_rect); pir; pir++)
  {
    avg = read_acc.read(DomainPoint::from_point<1>(pir.p));
  }
  double value;
  double sum  = 0.0;
  for (GenericPointInRectIterator<1> pir(write_rect); pir; pir++)
  {
    value = write_acc.read(DomainPoint::from_point<1>(pir.p));
    write_acc.write(DomainPoint::from_point<1>(pir.p), value - avg);
    sum += value - avg;
  }
  std::cout<<"Inside shift_task(), subtract average: "
	   <<avg<<std::endl;
}

void check_field_task(const Task *task,
		      const std::vector<PhysicalRegion> &regions,
		      Context ctx, HighLevelRuntime *runtime) {  

  std::cout<<"Inside check_field_task()"<<std::endl;
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, double> acc = 
    regions[0].get_field_accessor(fid).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();
  double sum  = 0.0;
  for (GenericPointInRectIterator<1> pir(rect); pir; pir++)
  {
    sum += acc.read(DomainPoint::from_point<1>(pir.p));
  }
  std::cout<<"check local sum: "<<sum<<std::endl;
}

int main(int argc, char *argv[]) {
  
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/,
      AUTO_GENERATE_ID, TaskConfigOptions(), "top_level");
  HighLevelRuntime::register_legion_task<spmd_detrend>(SPMD_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(), "spmd");
  HighLevelRuntime::register_legion_task<init_field_task>(INIT_FIELD_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true), "init");
  HighLevelRuntime::register_legion_task<local_sum_task>(LOCAL_SUM_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true), "local_sum");
  HighLevelRuntime::register_legion_task<compute_node_task>(COMPUTE_NODE_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true), "comp_node");
  HighLevelRuntime::register_legion_task<shift_task>(SHIFT_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true), "shift");
  HighLevelRuntime::register_legion_task<check_field_task>(CHECK_FIELD_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true), "check");  

  // register reduction operator
  Add::register_operator();

  
  return HighLevelRuntime::start(argc, argv);
}
