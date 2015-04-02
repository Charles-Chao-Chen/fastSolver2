#include "gemm.hpp"

// (specialized) matrix multiply:
//  note the reversed order in the result.
//  This is done with a special projector:
//   f(n) = is_even(n) ? n+1, n-1;
// -     -   -    -     -        -
// | V0' |   | u0 |     | V1'*u1 |
// |     | * |    |  =  |        |
// | V1' |   | u1 |     | V0'*u0 |
// -     -   -    -     -        -
void gemmRed() {


}
