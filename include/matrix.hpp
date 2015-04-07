#ifndef _matrix_hpp
#define _matrix_hpp

#include <vector>

class Vector {
public:
  Vector();
  Vector(int N);

  // number of rows
  int rows() const;

  // 2-norm
  double norm() const;

  // random entries
  void rand(int nPart);

  // entry-wise operations
  Vector multiply(const Vector&);
  friend Vector operator + (const Vector&, const Vector&);
  friend Vector operator - (const Vector&, const Vector&);
  
private:
  int nPart;
  int mRows;
  std::vector<int> seeds;
};

class Matrix {
public:
  Matrix();
  Matrix(int nRow, int nCol);

  // consistant with eigen routines
  int rows() const;
  int cols() const;

  // return the matrix transpose
  Matrix T();

  // matrix vector product
  friend Vector operator * (const Matrix&, const Vector&);
  
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
