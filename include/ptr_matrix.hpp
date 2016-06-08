#ifndef _ptr_matrix_hpp
#define _ptr_matrix_hpp

#include <iostream>
#include <string>

// The motivation is to encapsulate various matrix interpretation
//  from pointers.
// This matrix can work on data by existing pointers and provide
//  the wrapper for linear algebra operations with blas and lapack.
class PtrMatrix {
public:
  PtrMatrix();
  // allocate memory constructor
  // used in DenseBlcokTask
  PtrMatrix(int, int);
  // init with existing pointer
  PtrMatrix(int, int, int, double*, char trans='n');
  ~PtrMatrix();
  
  void rand(long seed, int offset=0);
  void display(const std::string&);

  int rows() const;
  int cols() const;
  int LD() const;
  double* pointer() const;
  double* pointer(int, int);

  void set_trans(char);
  
  void solve(PtrMatrix&);

  // set all entries to value
  void clear(double value);

  // scale all entries
  void scale(double value);
  
  // initialize to identity matrix
  void identity();

  // return entry/reference to the matrix entry
  double  operator()(int, int) const;
  double& operator()(int, int);

  static void add
  (double alpha, const PtrMatrix&,
   double beta,  const PtrMatrix&,
   PtrMatrix&);
  
  static void gemm
  (const PtrMatrix&, const PtrMatrix&, const PtrMatrix&, PtrMatrix&);
  
  static void gemm
  (double, const PtrMatrix&, const PtrMatrix&, PtrMatrix&);

  static void gemm
  (double, const PtrMatrix&, const PtrMatrix&, double, PtrMatrix&);

private:

  // private variables  
  int mRows;
  int mCols;
  int leadD; // leading dimension
  double *ptr;
  bool has_memory;
  
public:
  // 't' for transpose, 'n' for no transpose
  // used in gemm
  char trans; 
};

#endif
