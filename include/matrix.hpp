#ifndef _matrix_hpp
#define _matrix_hpp

#include <vector>
#include <string>

class Matrix;
class Vector {
public:
  Vector();
  Vector(int N, bool generate_entry=true);
  
  // number of rows
  int rows() const;

  // number of partitions
  int num_partition() const;
  
  // 2-norm
  double norm() const;

  // random entries
  //void rand(int nPart, bool generate_entry=true);
  void rand(int nPart);

  // return the random seed
  long rand_seed(int) const;
  
  // form a diagonal matrix
  Matrix to_diag_matrix() const;
  
  // return the ith entry / reference  
  double  operator[] (int i) const;
  double& operator[] (int i);
  
  // entry-wise operations
  Vector multiply(const Vector&);
  friend Vector operator + (const Vector&, const Vector&);
  friend Vector operator - (const Vector&, const Vector&);
  friend bool   operator== (const Vector&, const Vector&);
  friend bool   operator!= (const Vector&, const Vector&);
  friend Vector operator * (const double,  const Vector&);
  
  // for debugging purpose
  void display(const std::string&) const;

  // static methods
  template <int value>
  static Vector constant(int);
  
private:
  int nPart;
  int mRows;
  std::vector<long>   seeds;
  std::vector<double> data;

  // for large matrices, we can avoid generating the entries,
  //  but only store the seeds
  bool generate_entry;
};

class Matrix {
public:
  Matrix();
  Matrix(int nRow, int nCol, bool generate_entry=true);
  //  ~Matrix();

  // consistant with eigen routines
  int rows() const;
  int cols() const;

  // number of partitions
  int num_partition() const;
  
  // random matrix with a random seed for each partition
  // the partition is horizontal
  void rand(int nPart);

  // return the random seed
  long rand_seed(int) const;
  
  // assignment operator
  //  void operator= (const Matrix&);

  // return the entry / reference
  double operator() (int i, int j) const;
  double& operator() (int i, int j);

  // return the matrix transpose
  Matrix T();

  // matrix vector product
  friend Vector operator * (const Matrix&, const Vector&);
  friend Matrix operator * (const Matrix&, const Matrix&);
  friend Matrix operator + (const Matrix&, const Matrix&);
  friend Matrix operator - (const Matrix&, const Matrix&);
  friend bool   operator== (const Matrix&, const Matrix&);
  friend bool   operator!= (const Matrix&, const Matrix&);
  friend Matrix operator * (const double,  const Matrix&);

  // for debugging purpose
  void display(const std::string&) const;

  // static methods
  template <int value>
  static Matrix constant(int m, int n);
  
private:
  int  nPart;
  int  mRows;
  int  mCols;
  std::vector<long>   seeds;
  std::vector<double> data;

  // for large matrices, we can avoid generating the entries,
  //  but only store the seeds
  bool generate_entry;
};

template <int value>
Vector Vector::constant(int N) {
  Vector temp(N);
  for (int i=0; i<N; i++)
    temp[i] = value;
  return temp;
}

template <int value>
Matrix Matrix::constant(int m, int n) {
  Matrix temp(m, n);
  for (int i=0; i<m; i++)
    for (int j=0; j<n; j++)
      temp(i, j) = value;
  return temp;
}

#endif
