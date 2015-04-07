#include "matrix.hpp"

#include <assert.h>

Vector::Vector() {}

Vector::Vector(int N) : mRows(N) {}

int Vector::rows() const {return mRows;}

double Vector::norm() const {
  return -1;
}

void Vector::rand(int nPart) {

}

double& Vector::operator[] (int i) {
  assert( 0 < i && i < mRows );
  double* temp = new double;
  *temp = -1;
  double& ref = *temp;
  return ref;
}

double Vector::operator[] (int i) const {
  assert( 0 < i && i < mRows );
  return -1;
}

Vector Vector::multiply(const Vector& other) {
  assert( this->rows() == other.rows() );
  Vector temp;
  for (int i=0; i<mRows; i++)
    temp[i] = (*this)[i] * other[i];
  return temp;
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

Matrix::Matrix(int row, int col) : mRows(row), mCols(col) {}

int Matrix::rows() const {return mRows;}

int Matrix::cols() const {return mCols;}

void Matrix::rand(int nPart) {

}

Matrix Matrix::T() { return Matrix(); }

Vector operator * (const Matrix& A, const Vector& b) {
  return b;
}

void Matrix::display() const {

}

void Matrix::destroy() {

}
