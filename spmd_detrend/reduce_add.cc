#include "reduce_add.hpp"

const ReductionOpID REDOP_ADD = 4321;

const double Add::identity = 0.0;

template<>
void Add::apply<true>(LHS &lhs, RHS rhs)
{
  lhs += rhs;
}

template<>
void Add::apply<false>(LHS &lhs, RHS rhs){
  
  int64_t *target = (int64_t *)&lhs;
  union { int64_t as_int; double as_T; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_T = oldval.as_T + rhs;
  } while (!__sync_bool_compare_and_swap(target,
					 oldval.as_int,
					 newval.as_int)
	   );
}

template<>
void Add::fold<true>(RHS &rhs1, RHS rhs2)
{
  rhs1 += rhs2;
}

template<>
void Add::fold<false>(RHS &rhs1, RHS rhs2)
{
  int64_t *target = (int64_t *)&rhs1;
  union { int64_t as_int; double as_T; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_T = oldval.as_T + rhs2;
  } while (!__sync_bool_compare_and_swap(target,
					 oldval.as_int,
					 newval.as_int)
	   );
}

void Add::register_operator() {
  HighLevelRuntime::register_reduction_op<Add>(REDOP_ADD);
}
