#ifndef _lmatrix_hpp
#define _lmatrix_hpp

#include <string>

#include "legion.h"
using namespace LegionRuntime::HighLevel;

#include "utility.hpp" // for WAIT_DEFAULT and FIELDID_V
#include "matrix.hpp"
#include "solver_tasks.hpp"

// legion matrix
class LMatrix {
public:
  LMatrix();
  LMatrix(int, int, int, Context, HighLevelRuntime*);
  LMatrix(IndexPartition ip, LogicalRegion lr, Context, HighLevelRuntime*);
  ~LMatrix();
  
  int rows() const;
  int cols() const;
  int num_partition() const;
  Domain color_domain() const;
  LogicalRegion logical_region() const;
  LogicalPartition logical_partition() const;

  // create logical region
  void create(int, int, Context, HighLevelRuntime*,
	      bool wait=WAIT_DEFAULT);

  // initialize region, assuming it exists
  /*
  void init_data
  (int, const Vector& Rhs, Context, HighLevelRuntime*,
   bool wait=WAIT_DEFAULT);
*/

  // set the matrix to value
  void clear(double, Context, HighLevelRuntime*, bool wait=WAIT_DEFAULT);

  // scale all the entries
  void scale(double, Context, HighLevelRuntime*, bool wait=WAIT_DEFAULT);
  
  // for UTree init and VTree init
  void init_data
  (int, int, int, const Matrix& VMat, Context, HighLevelRuntime*,
   bool wait=WAIT_DEFAULT);

  // for KTree::partition
  void init_dense_blocks
  (int, int, const Matrix& UMat, const Matrix& VMat, const Vector& DVec,
   Context, HighLevelRuntime*, bool wait=WAIT_DEFAULT);

  // uniform partition
  // Note: the first argument is level, NOT nPart
  void partition(int level, Context, HighLevelRuntime*);

  // return a new matrix with created partition,
  //  so no need to destroy region twice
  LMatrix partition
  (int level, int col0, int col1,
   Context ctx, HighLevelRuntime *runtime);
  
  // output the right hand side
  // for UTree::rhs()
  Vector to_vector();

  // for KTree::partition()
  //void create_dense_partition
  //(int, const Matrix& U, const Matrix& V, const Vector& D,
  //Context, HighLevelRuntime*, bool wait=WAIT_DEFAULT);
  
  // solve linear system
  // for KTree::solve()
  void solve
  (LMatrix&, Context, HighLevelRuntime*, bool wait=WAIT_DEFAULT);

  // solve node system
  // for HMatrix::solve()
  void node_solve
  (LMatrix&, Context, HighLevelRuntime*, bool wait=WAIT_DEFAULT);

  // print the values on screen
  // for debugging
  void display(const std::string&,
	       Context, HighLevelRuntime*, bool wait=WAIT_DEFAULT);

  // static methods
  // matrix subtraction
  static void add
  (double alpha, const LMatrix&,
   double beta,  const LMatrix&, LMatrix&,
   Context, HighLevelRuntime*, bool wait=WAIT_DEFAULT);

  // gemm reduction
  // C = alpha*op(A) * op(B) + beta*C
  static void gemmRed
  (char, char, double, const LMatrix&, const LMatrix&,
   double, LMatrix&, Context, HighLevelRuntime*,
   bool wait=WAIT_DEFAULT);

  // gemm broadcast
  static void gemmBro
  (double, const LMatrix&, const LMatrix&,
   double, const LMatrix&, Context, HighLevelRuntime*,
   bool wait=WAIT_DEFAULT);
  
private:

  // ******************
  // helper functions
  // ******************

  // for init_matrix task
  ArgumentMap MapSeed(int nPart, const Matrix& matrix);
  ArgumentMap MapSeed(int nPart, const Matrix& U, const Matrix& V, const Vector& D);

  // partition the matrix along rows
  IndexPartition UniformRowPartition
  (int num_subregions, int, int, Context ctx, HighLevelRuntime *runtime);
  
  // for node_solve
  void coarse_partition();
  void fine_partition();

  template <typename T>
  void solve
  (LMatrix& b, Context ctx, HighLevelRuntime *runtime,
   bool wait=WAIT_DEFAULT);

  // ******************
  // private variables
  // ******************
  
  // matrix and block size
  int mRows;
  int mCols;
  int rblock;
  //int cblock;

  // number of ranks
  // used to init data
  int              nProc;  

  // partition
  int              nPart;
  Domain           colDom;
  IndexPartition   ipart;
  LogicalPartition lpart;

  // region
  IndexSpace       ispace;
  FieldSpace       fspace;
  LogicalRegion    region;
};

#endif
