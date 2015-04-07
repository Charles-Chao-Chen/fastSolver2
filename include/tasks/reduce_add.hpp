#ifndef _reduce_add_hpp
#define _reduce_add_hpp

#include "legion.h"
using namespace LegionRuntime::HighLevel;

extern const int REDOP_ADD;

// Reduction Op
class Add {
public:
  typedef double LHS;
  typedef double RHS;
  static const double identity;

public:
  template <bool EXCLUSIVE>
  static void apply(LHS &lhs, RHS rhs);
  template <bool EXCLUSIVE>
  static void fold(RHS &rhs1, RHS rhs2);
  static void register_operator();
};

#endif
