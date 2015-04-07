#ifndef _matrix_hpp
#define _matrix_hpp

#include <vector>
#include <string>

class Vector {
public:
  Vector();
  Vector(int N);
  ~Vector();
  
  // number of rows
  int rows() const;

  // 2-norm
  double norm() const;

  // random entries
  void rand(int nPart);

  // return the ith entry / reference
  double& operator[] (int i);
  double  operator[] (int i) const;
  
  // entry-wise operations
  Vector multiply(const Vector&);
  friend Vector operator + (const Vector&, const Vector&);
  friend Vector operator - (const Vector&, const Vector&);

  // for debugging purpose
  void display(const std::string&) const;
  
private:
  int nPart;
  int mRows;
  double *data;
  std::vector<long> seeds;
};

class Matrix {
public:
  Matrix();
  Matrix(int nRow, int nCol);
  ~Matrix();

  // consistant with eigen routines
  int rows() const;
  int cols() const;

  // random matrix with a random seed for each partition
  // the partition is horizontal
  void rand(int nPart);

  // return the entry / reference
  double operator() (int i, int j) const;
  double& operator() (int i, int j);

  // return the matrix transpose
  Matrix T();

  // matrix vector product
  friend Vector operator * (const Matrix&, const Vector&);
  
  // print the values on screen
  void display(const std::string&) const;
  
private:
  int nPart;
  int mRows;
  int mCols;
  double *data;
  std::vector<long> seeds;
};

#endif
