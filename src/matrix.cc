#include "matrix.hpp"
#include "ptr_matrix.hpp"
#include "lapack_blas.hpp"

#include <iostream>
#include <assert.h>
#include <math.h>   // for sqrt()
#include <stdlib.h> // for srand48_r(), lrand48_r() and drand48_r()
#include <time.h>

Vector::Vector() : nPart(-1), mRows(-1), has_entry(true) {}

Vector::Vector(int N, bool has) : nPart(-1), mRows(N), has_entry(has) {
  assert(N>0);
  if (has_entry) {
    // allocate memory
    data.resize(mRows);
  }
}

int Vector::rows() const {return mRows;}

int Vector::num_partition() const {return nPart;}

long Vector::rand_seed(int i) const {
  assert( 0<=i && i<nPart );
  return seeds[i];
}

double Vector::norm() const {
  assert( has_entry == true );
  double sum = 0.0;
  for (int i=0; i<mRows; i++)
    sum += data[i]*data[i];
  return sqrt(sum);
}

void Vector::rand(int nPart_) {
  assert( mRows%nPart == 0 );
  this->nPart = nPart_;
  struct drand48_data buffer;
  assert( srand48_r( time(NULL)+2, &buffer ) == 0 );
  for (int i=0; i<nPart; i++) {
    long seed;
    assert( lrand48_r(&buffer, &seed) == 0 );
    seeds.push_back( seed );
  }

  // generating random numbers
  if (has_entry) {
    int count = 0;
    int colorSize = mRows / nPart;
    for (int i=0; i<nPart; i++) {
      assert( srand48_r( seeds[i], &buffer ) == 0 );
      for (int j=0; j<colorSize; j++) {
	assert( drand48_r(&buffer, &data[count]) == 0 );
	count++;
      }
    }
  }
}

double& Vector::operator[] (int i) {
  assert( has_entry == true );
  assert( 0 <= i && i < mRows );
  return data[i];
}

double Vector::operator[] (int i) const {
  assert( has_entry == true );
  assert( 0 <= i && i < mRows );
  return data[i];
}

Matrix Vector::to_diag_matrix() const {
  Matrix temp = Matrix::constant<0>( mRows, mRows );
  for (int i=0; i<mRows; i++)
    temp(i, i) = data[i];
  return temp;
}

Vector Vector::multiply(const Vector& other) {
  assert( this->has_entry == true );
  assert( other.has_entry == true );
  int N = this->rows();
  assert( N == other.rows() );
  Vector temp(N);
  for (int i=0; i<N; i++)
    temp[i] = data[i] * other[i];
  return temp;
}

void Vector::display(const std::string& name) const {
  std::cout << name << ":" << std::endl;
  if (nPart > 0) {
    std::cout << "seeds:" << std::endl;
    for (int i=0; i<nPart; i++)
      std::cout << seeds[i] << "\t";
    std::cout << std::endl
	      << "values:"
	      << std::endl;
  }
  if (has_entry) {
    for (int j=0; j<mRows; j++)
      std::cout << data[j] << "\t";
    std::cout << std::endl;
  }
}

Vector operator + (const Vector& vec1, const Vector& vec2) {
  assert( vec1.has_entry == true );
  assert( vec2.has_entry == true );
  assert( vec1.rows() == vec2.rows() );
  Vector temp(vec1.rows());
  for (int i=0; i<vec1.rows(); i++)
    temp[i] = vec1[i] + vec2[i];
  return temp;
}

Vector operator - (const Vector& vec1, const Vector& vec2) {
  assert( vec1.has_entry == true );
  assert( vec2.has_entry == true );
  assert( vec1.rows() == vec2.rows() );
  Vector temp(vec1.rows());
  for (int i=0; i<vec1.rows(); i++)
    temp[i] = vec1[i] - vec2[i];
  return temp;
}

bool operator== (const Vector& vec1, const Vector& vec2) {  
  assert( vec1.has_entry == true );
  assert( vec2.has_entry == true );
  assert( vec1.rows() == vec2.rows() );
  for (int i=0; i<vec1.rows(); i++) {
    if ( fabs(vec1[i] - vec2[i]) > 1e-10 )
      return false;
  }
  return true;
}

bool operator!= (const Vector& vec1, const Vector& vec2) {
  return !(vec1 == vec2);
}

Vector operator * (const double alpha,  const Vector& vec) {
  assert( vec.has_entry == true );
  Vector temp(vec.rows());
  for (int i=0; i<vec.rows(); i++)
    temp[i] = alpha * vec[i];
  return temp;
}

Matrix::Matrix() : nPart(-1), mRows(-1), mCols(-1), has_entry(true) {}

Matrix::Matrix(int row, int col, bool has)
  : nPart(-1), mRows(row), mCols(col), has_entry(has) {
  
  assert( mRows>0 && mCols>0 );
  if (has_entry) {
    // allocate memory
    data.resize(mRows*mCols);
  }
}

int Matrix::rows() const {return mRows;}

int Matrix::cols() const {return mCols;}

double* Matrix::pointer() {return &data[0];}

int Matrix::num_partition() const {return nPart;}

void Matrix::rand(int nPart_) {
  this->nPart = nPart_;
  assert( mRows%nPart == 0 );
  struct drand48_data buffer;
  assert( srand48_r( time(NULL) + lrand48(), &buffer ) == 0 );
  for (int i=0; i<nPart; i++) {
    long seed;
    assert( lrand48_r(&buffer, &seed) == 0 );
    seeds.push_back( seed );
  }
    
  // generating random numbers
  if (has_entry) {
    int nrow = mRows / nPart;
    for (int k=0; k<nPart; k++) {
      PtrMatrix pMat(nrow, mCols, mRows, &data[k*nrow]);
      pMat.rand( seeds[k] );
    }
  }
}

long Matrix::rand_seed(int i) const {
  assert( 0<=i && i<nPart );
  return seeds[i];
}

double Matrix::operator() (int i, int j) const {
  assert( has_entry == true );
  return data[i+j*mRows];
}

double& Matrix::operator() (int i, int j) {
  assert( has_entry == true );
  return data[i+j*mRows];
}

Matrix Matrix::T() {
  assert( has_entry == true );
  Matrix temp(mCols, mRows);
  for (int i=0; i<mRows; i++)
    for (int j=0; j<mCols; j++)
      temp(j, i) = (*this)(i, j);
  return temp;
}

Matrix Matrix::block(int rlo, int rhi, int clo, int chi) const {
  assert(rhi>rlo && chi>clo);
  Matrix temp(rhi-rlo, chi-clo);
  for (int i=0; i<temp.rows(); i++)
    for (int j=0; j<temp.cols(); j++)
      temp(i, j) = (*this)(rlo+i, clo+j);
  return temp;
}

void Matrix::solve(Matrix &B) {
  int N = this->mRows;
  int NRHS = B.cols();
  int LDA = mRows;
  int LDB = B.rows();
  int IPIV[N];
  int INFO;
  lapack::dgesv_(&N, &NRHS, this->pointer(), &LDA, IPIV,
		 B.pointer(), &LDB, &INFO);
  assert(INFO==0);  
}

Vector operator * (const Matrix& A, const Vector& b) {
  assert( A.has_entry == true );
  assert( A.cols() == b.rows() );
  Vector temp(A.rows());
  for (int i=0; i<temp.rows(); i++) {
    temp[i] = 0.0;
    for (int j=0; j<A.cols(); j++)
      temp[i] += A(i,j) * b[j];
  }
  return temp;
}

Matrix operator + (const Matrix& mat1, const Matrix& mat2) {
  assert( mat1.has_entry == true );
  assert( mat2.has_entry == true );
  assert(mat1.rows() == mat2.rows());
  assert(mat1.cols() == mat2.cols());
  Matrix temp(mat1.rows(), mat1.cols());
  for (int i=0; i<temp.rows(); i++)
    for (int j=0; j<temp.cols(); j++)
      temp(i, j) = mat1(i, j) + mat2(i, j);
  return temp;
}

Matrix operator - (const Matrix& mat1, const Matrix& mat2) {
  assert( mat1.has_entry == true );
  assert( mat2.has_entry == true );
  assert(mat1.rows() == mat2.rows());
  assert(mat1.cols() == mat2.cols());
  Matrix temp(mat1.rows(), mat1.cols());
  for (int i=0; i<temp.rows(); i++)
    for (int j=0; j<temp.cols(); j++)
      temp(i, j) = mat1(i, j) - mat2(i, j);
  return temp;
}

bool operator== (const Matrix& mat1, const Matrix& mat2) {
  assert( mat1.has_entry == true );
  assert( mat2.has_entry == true );
  assert(mat1.rows() == mat2.rows());
  assert(mat1.cols() == mat2.cols());
  for (int i=0; i<mat1.rows(); i++)
    for (int j=0; j<mat1.cols(); j++)
      if ( fabs( mat1(i, j) - mat2(i, j) ) > 1.0e-10 )
	return false;
  return true;
}

bool operator!= (const Matrix& mat1, const Matrix& mat2) {
  return !(mat1 == mat2);
}

Matrix operator * (const double alpha,  const Matrix& mat) {
  assert( mat.has_entry == true );
  Matrix temp(mat.rows(), mat.cols());
  for (int i=0; i<temp.rows(); i++)
    for (int j=0; j<temp.cols(); j++)
      temp(i, j) = alpha * mat(i, j);
  return temp;
}

Matrix operator * (const Matrix& mat1, const Matrix& mat2) {
  assert(mat1.cols() == mat2.rows());
  Matrix temp( mat1.rows(), mat2.cols());
  for (int i=0; i<temp.rows(); i++) {
    for (int j=0; j<temp.cols(); j++) {
      temp(i, j) = 0.0;
      for (int k=0; k<mat1.cols(); k++)
	temp(i, j) += mat1(i, k) * mat2(k, j);
    }
  }
  return temp;
}

// the fastest increasing dimension is displayed first
void Matrix::display(const std::string& name) const {
  std::cout << name << ":" << std::endl;
  if (nPart > 0) {
    std::cout << "seeds:" << std::endl;
    for (int i=0; i<nPart; i++)
      std::cout << seeds[i] << "\t";
    std::cout << std::endl
	      << "values:"
	      << std::endl;
  }
  if (has_entry) {
    for (int i=0; i<mRows; i++) {
      for (int j=0; j<mCols; j++)
	std::cout << (*this)(i, j) << "\t";
      std::cout << std::endl;
    }
  }
}

Matrix Matrix::identity(int N) {
  Matrix temp = Matrix::constant<0>(N, N);
  for (int i=0; i<N; i++)
    temp(i, i) = 1.0;
  return temp;
}
