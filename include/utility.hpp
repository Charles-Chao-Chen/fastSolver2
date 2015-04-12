#ifndef _utility_hpp
#define _utility_hpp

#include "legion.h"
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

enum {
  FIELDID_V,
};

const bool WAIT_DEFAULT = true; // waiting for tasks

double* region_pointer(const PhysicalRegion &region, int, int, int, int);

// error message
#include <cstdlib> // for EXIT_FAILURE
#include <cassert>
#define Error(msg) {				\
    std::cerr					\
      << "Error in file : " << __FILE__		\
      << ", function : " << __func__		\
      << ", line : " << __LINE__		\
      << "\n\t" << msg << std::endl;		\
    assert(false);				\
  }
  //exit(EXIT_FAILURE)

#endif
