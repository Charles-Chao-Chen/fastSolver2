#ifndef _ptr_matrix_hpp
#define _ptr_matrix_hpp

#include <iostream>
#include <string>

// The motivation is to encapsulate various matrix interpretation
//  from pointers.
// This matrix works on data by existing pointers and does not deal
//   with any memory allocation or destruction.
// The linear algebra operations rely on blas and lapack.
class PtrMatrix {
public:
  PtrMatrix();
  PtrMatrix(int, int, int, double*);
  void rand(long seed);
  void display(const std::string&);
  
  //friend std::ostream& operator<< (std::ostream& stream, const PtrMatrix&);

private:
  // helper function
  // return the pointer to the matrix entry
  double* operator()(int, int);

  
  int mRows;
  int mCols;
  int LeadD; // leading dimension
  double *ptr;
};

#endif
