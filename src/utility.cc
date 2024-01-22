#include "utility.hpp"

bool is_power_of_two(int x) {
  return (x > 0) && !(x & (x-1));
}

double* region_pointer
(const PhysicalRegion &region, int rlo, int rhi, int clo, int chi) {
  Rect<2> bounds, subrect;
  bounds.lo[0] = rlo;
  bounds.hi[0] = rhi-1;
  bounds.lo[1] = clo;
  bounds.hi[1] = chi-1;
  FieldAccessor<LEGION_READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t> > accessor(region, FIELDID_V);
  size_t offset[2];
  double *base = accessor.ptr(bounds, offset, 2);
  assert(offset[0] == sizeof(double));
#ifdef DEBUG_POINTERS
  printf("ptr = %p (%d, %d)\n", base, offset[0], offset[1]);
#endif
  return base;
}

PtrMatrix get_raw_pointer
(const PhysicalRegion &region, int rlo, int rhi, int clo, int chi,
 bool wait) {
  Rect<2> bounds, subrect;
  bounds.lo[0] = rlo;
  bounds.hi[0] = rhi-1;
  bounds.lo[1] = clo;
  bounds.hi[1] = chi-1;
  FieldAccessor<LEGION_READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t> > accessor(region, FIELDID_V);
  size_t offset[2];
  double *base = accessor.ptr(bounds, offset, 2);
  assert(offset[0] == sizeof(double));
  int ld = offset[1]/sizeof(double);
  assert(ld>=rhi-rlo);
  return PtrMatrix(rhi-rlo, chi-clo, ld, base);
}

PtrMatrix reduction_pointer
(const PhysicalRegion &region, int rlo, int rhi, int clo, int chi) {
  Rect<2> bounds, subrect;
  bounds.lo[0] = rlo;
  bounds.hi[0] = rhi-1;
  bounds.lo[1] = clo;
  bounds.hi[1] = chi-1;
  FieldAccessor<LEGION_READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t> > accessor(region, FIELDID_V);
  size_t offset[2];
  double *base = accessor.ptr(bounds, offset, 2);
  assert(offset[0] == sizeof(double));
#ifdef DEBUG_POINTERS
  printf("ptr = %p (%d, %d)\n", base, offset[0], offset[1]);
#endif
  int ld = offset[1]/sizeof(double);
  return PtrMatrix(rhi-rlo, chi-clo, ld, base);
}
