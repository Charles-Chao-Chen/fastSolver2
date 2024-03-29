#include "utility.hpp"
#include "tasks/reduce_add.hpp"

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
  double *base = accessor.ptr(bounds, offset);
#ifdef DEBUG_POINTERS
  printf("ptr = %p (%d, %d)\n", base, offset[0]*sizeof(double), offset[1]*sizeof(double));
#endif
  return base;
}

template<PrivilegeMode PRIVILEGE>
PtrMatrix get_raw_pointer
(const PhysicalRegion &region, int rlo, int rhi, int clo, int chi) {
  Rect<2> bounds, subrect;
  bounds.lo[0] = rlo;
  bounds.hi[0] = rhi-1;
  bounds.lo[1] = clo;
  bounds.hi[1] = chi-1;
  FieldAccessor<PRIVILEGE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t> > accessor(region, FIELDID_V);
  size_t offset[2];
  double *base = const_cast<double *>(accessor.ptr(bounds, offset));
  int ld = offset[1];
  assert(ld>=rhi-rlo);
  return PtrMatrix(rhi-rlo, chi-clo, ld, base);
}
template PtrMatrix get_raw_pointer<LEGION_READ_ONLY>(const PhysicalRegion &, int, int, int, int);
template PtrMatrix get_raw_pointer<LEGION_READ_WRITE>(const PhysicalRegion &, int, int, int, int);

PtrMatrix reduction_pointer
(const PhysicalRegion &region, int rlo, int rhi, int clo, int chi) {
  Rect<2> bounds, subrect;
  bounds.lo[0] = rlo;
  bounds.hi[0] = rhi-1;
  bounds.lo[1] = clo;
  bounds.hi[1] = chi-1;
  ReductionAccessor<Add,true,2,coord_t,Realm::AffineAccessor<double,2,coord_t> > accessor(region, FIELDID_V, REDOP_ADD);
  size_t offset[2];
  double *base = accessor.ptr(bounds, offset);
#ifdef DEBUG_POINTERS
  printf("ptr = %p (%d, %d)\n", base, offset[0]*sizeof(double), offset[1]*sizeof(double));
#endif
  int ld = offset[1];
  return PtrMatrix(rhi-rlo, chi-clo, ld, base);
}
