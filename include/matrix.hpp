#ifndef _matrix_hpp
#define _matrix_hpp

#include "vector.h"

class Matrix {
public:
  Matrix();

  // consistant with eigen routines
  int rows();
  int cols();

  // random matrix with a random seed for each partition
  // the partition is horizontal
  void rand(int nPart);

  // print the values on screen
  void display();
  
private:
  int nPart;
  std::vector<int> seeds;
};

#endif
