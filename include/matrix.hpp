#ifndef _matrix_hpp
#define _matrix_hpp

#include "vector.h"

class Vector;

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

  // destructor
  void destroy();
  
private:
  int nPart;
  std::vector<int> seeds;
};

#endif
