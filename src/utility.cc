#include "utility.hpp"

double* region_pointer
(const PhysicalRegion &region, int rlo, int rhi, int clo, int chi) {
  Rect<2> bounds, subrect;
  bounds.lo.x[0] = rlo;
  bounds.hi.x[0] = rhi-1;
  bounds.lo.x[1] = clo;
  bounds.hi.x[1] = chi-1;
  ByteOffset offsets[2];
  double *base = region.get_field_accessor(FIELDID_V).template typeify<double>().template raw_rect_ptr<2>(bounds, subrect, offsets);
  assert(subrect == bounds);
#ifdef DEBUG_POINTERS
  printf("ptr = %p (%d, %d)\n", base, offsets[0].offset, offsets[1].offset);
#endif
  return base;
}
