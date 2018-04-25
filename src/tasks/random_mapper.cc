//#include "utilities.h"

#include "random_mapper.hpp"
#include <algorithm>    // std::shuffle
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

using namespace LegionRuntime;

//Realm::Logger log_solver_mapper("random_mapper");

RandomMapper::RandomMapper(Machine machine, Runtime *rt, Processor local,
			   std::vector<Processor>* _procs_list,
			   std::vector<Memory>* _sysmems_list,
			   std::map<Memory, std::vector<Processor> >* _sysmem_local_procs,
			   std::map<Processor, Memory>* _proc_sysmems,
			   std::map<Processor, Memory>* _proc_regmems)
  : DefaultMapper(rt->get_mapper_runtime(), machine, local),
    procs_list(*_procs_list),
    sysmems_list(*_sysmems_list),
    sysmem_local_procs(*_sysmem_local_procs),
    proc_sysmems(*_proc_sysmems),
  proc_regmems(*_proc_regmems)
{}


void RandomMapper::slice_task(const MapperContext      ctx,
			      const Task&              task, 
			      const SliceTaskInput&    input,
			      SliceTaskOutput&   output) {

  default_slice_task(task, local_cpus, remote_cpus, 
		     input, output, cpu_slices_cache);
}
      
void RandomMapper::default_slice_task(const Task &task,
				      const std::vector<Processor> &local,
				      const std::vector<Processor> &remote,
				      const SliceTaskInput& input,
				      SliceTaskOutput &output,
				      std::map<Domain,std::vector<TaskSlice> >
				      &cached_slices) const {
  
  // Before we do anything else, see if it is in the cache
  std::map<Domain,std::vector<TaskSlice> >::const_iterator finder = 
    cached_slices.find(input.domain);
  if (finder != cached_slices.end()) {
    output.slices = finder->second;
    return;
  }

  Machine::ProcessorQuery all_procs(machine);
  all_procs.only_kind(local[0].kind());
  std::vector<Processor> procs(all_procs.begin(), all_procs.end());

  // obtain a time-based seed:
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  shuffle(procs.begin(), procs.end(), std::default_random_engine(seed));

  assert(input.domain.get_dim() == 1);
  Arrays::Rect<1> point_rect = input.domain.get_rect<1>();
  Arrays::Point<1> num_blocks(point_rect.volume());
  default_decompose_points<1>(point_rect, procs,
			      num_blocks, false/*recurse*/,
			      stealing_enabled, output.slices);  
}

