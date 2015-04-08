#include "matrix.hpp"

#include <iostream>
#include <assert.h>
#include <math.h>   // for sqrt()
#include <stdlib.h> // for srand48_r(), lrand48_r() and drand48_r()
#include <time.h>

Vector::Vector() {}

Vector::Vector(int N) : mRows(N) {
  data = new double[mRows];
  assert( data != NULL );
}

Vector::~Vector() {
  if (data != NULL) {
    delete[] data;
    data = NULL;
  }
}

int Vector::rows() const {return mRows;}

int Vector::num_partition() const {return nPart;}

long Vector::rand_seed(int i) const {
  assert( 0<=i && i<nPart );
  return seeds[i];
}

double Vector::norm() const {
  double sum = 0.0;
  for (int i=0; i<mRows; i++)
    sum += data[i]*data[i];
  return sqrt(sum);
}

void Vector::rand(int nPart_) {
  this->nPart = nPart_;
  assert( mRows%nPart == 0 );
  int count = 0;
  int colorSize = mRows / nPart;
  struct drand48_data buffer;
  assert( srand48_r( time(NULL), &buffer ) == 0 );
  for (int i=0; i<nPart; i++) {
    long seed;
    assert( lrand48_r(&buffer, &seed) == 0 );
    seeds.push_back( seed );
    for (int j=0; j<colorSize; j++) {
      assert( drand48_r(&buffer, &data[count]) == 0 );
      count++;
    }
  }
}

double& Vector::operator[] (int i) {
  assert( 0 <= i && i < mRows );
  return data[i];
}

double Vector::operator[] (int i) const {
  assert( 0 <= i && i < mRows );
  return data[i];
}

Vector Vector::multiply(const Vector& other) {
  assert( this->rows() == other.rows() );
  Vector temp;
  for (int i=0; i<mRows; i++)
    temp[i] = data[i] * other[i];
  return temp;
}

void Vector::display(const std::string& name) const {
  std::cout << name << ":" << std::endl;
  int count = 0;
  int colorSize = mRows / nPart;
  for (int i=0; i<nPart; i++) {
    std::cout << "seed=" << seeds[i] << std::endl;
    for (int j=0; j<colorSize; j++) {
      std::cout << data[count] << "\t";
      count++;
    }
    std::cout << std::endl;
  }
}

Vector operator + (const Vector& vec1, const Vector& vec2) {
  assert( vec1.rows() == vec2.rows() );
  Vector temp;
  for (int i=0; i<vec1.rows(); i++)
    temp[i] = vec1[i] + vec2[i];
  return temp;
}

Vector operator - (const Vector& vec1, const Vector& vec2) {
  assert( vec1.rows() == vec2.rows() );
  Vector temp;
  for (int i=0; i<vec1.rows(); i++)
    temp[i] = vec1[i] - vec2[i];
  return temp;
}

Matrix::Matrix() {}

Matrix::Matrix(int row, int col) : mRows(row), mCols(col) {
  data = new double[mRows*mCols];
  assert( data != NULL );
}

Matrix::~Matrix() {
  if (data != NULL) {
    delete[] data;
    data = NULL;
  }
}

int Matrix::rows() const {return mRows;}

int Matrix::cols() const {return mCols;}

int Matrix::num_partition() const {return nPart;}

void Matrix::rand(int nPart_) {
  this->nPart = nPart_;
  assert( mRows%nPart == 0 );
  int count = 0;
  int colorSize = mRows / nPart;
  struct drand48_data buffer;
  assert( srand48_r( time(NULL), &buffer ) == 0 );
  for (int i=0; i<nPart; i++) {
    long seed;
    assert( lrand48_r(&buffer, &seed) == 0 );
    seeds.push_back( seed );
    for (int j=0; j<colorSize*mCols; j++) {
      assert( drand48_r(&buffer, &data[count]) == 0 );
      count++;
    }
  }  
}

long Matrix::rand_seed(int i) const {
  assert( 0<=i && i<nPart );
  return seeds[i];
}

double Matrix::operator() (int i, int j) const {
  return data[i+j*mRows];
}

double& Matrix::operator() (int i, int j) {
  return data[i+j*mRows];
}

Matrix Matrix::T() {
  Matrix temp;
  for (int i=0; i<mRows; i++)
    for (int j=0; j<mCols; j++)
      temp(j, i) = (*this)(i, j);
  return temp;
}

Vector operator * (const Matrix& A, const Vector& b) {
  assert( A.cols() == b.rows() );
  Vector temp(A.rows());
  for (int  j=0; j<A.rows(); j++) {
    temp[j] = 0.0;
    for (int i=0; i<A.cols(); i++)
      temp[j] += A(i,j) * b[i];
  }
  return temp;
}

// the fastest increasing dimension is displayed first
void Matrix::display(const std::string& name) const {
  std::cout << name << ":" << std::endl;
  int count = 0;
  int colorSize = mRows / nPart;
  for (int i=0; i<nPart; i++) {
    std::cout << "seed=" << seeds[i] << std::endl;
    for (int j=0; j<colorSize*mCols; j++) {
      std::cout << data[count] << "\t";
      count++;
    }
    std::cout << std::endl;
  }
}
