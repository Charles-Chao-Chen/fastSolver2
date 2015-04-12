#ifndef _macros_hpp
#define _macros_hpp

enum {
  FIELDID_V,
};

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

const bool WAIT_DEFAULT = true; // waiting for tasks

#endif
