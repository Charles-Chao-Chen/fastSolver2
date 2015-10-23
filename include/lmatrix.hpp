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
  /*
  LMatrix
  (int, int, int, IndexPartition ip, LogicalRegion lr,
   Context, HighLevelRuntime*); // to be removed
  */
  ~LMatrix();
  
  int rows() const;
  int cols() const;
  int rowBlk() const;
  int column_begin() const;
  int num_partition() const;
  int partition_level() const;
  Domain color_domain() const;
  IndexSpace index_space() const;
  LogicalRegion logical_region() const;
  IndexPartition index_partition() const;
  LogicalPartition logical_partition() const;

  void set_column_size(int);
  void set_column_begin(int);
  void set_logical_region(LogicalRegion);
  void set_parent_region(LogicalRegion);
  void set_logical_partition(LogicalPartition lp);
  
  // create logical region
  void create(int, int, Context, HighLevelRuntime*);

  // set the matrix to value
  void clear
  (double, Context, HighLevelRuntime*, bool wait=WAIT_DEFAULT);

  // scale all the entries
  void scale
  (double, Context, HighLevelRuntime*, bool wait=WAIT_DEFAULT);

  // initialize data from Matrix object
  void init_data
  (const Matrix& mat, Context, HighLevelRuntime*,
   bool wait=WAIT_DEFAULT);

  // init part of the region
  void init_data
  (int, int, const Matrix& mat, Context, HighLevelRuntime*,
   bool wait=WAIT_DEFAULT);
  
  // output region
  Matrix to_matrix(Context, HighLevelRuntime*);
  Matrix to_matrix(int, int, Context, HighLevelRuntime*);
  Matrix to_matrix(int, int, int, int, Context, HighLevelRuntime*);

  // to be removed
  void init_data
  (int, const Matrix& VMat, Context, HighLevelRuntime*,
   bool wait=WAIT_DEFAULT);

  // specifies column range
  // for UTree init
  void init_data
  (int, int, int, const Matrix& VMat, Context, HighLevelRuntime*,
   bool wait=WAIT_DEFAULT);
  
  // for KTree::partition
  void init_dense_blocks
  (const Matrix& UMat, const Matrix& VMat, const Vector& DVec,
   Context, HighLevelRuntime*, bool wait=WAIT_DEFAULT);

  void init_dense_blocks
  (int, int, const Matrix& UMat, const Matrix& VMat, const Vector& DVec,
   Context, HighLevelRuntime*, bool wait=WAIT_DEFAULT);

  // uniform partition
  // Note: the first argument is level, NOT nPart
  void partition(int level, Context, HighLevelRuntime*);

  /*
  // return a new matrix with created partition,
  //  so no need to destroy region twice
  LMatrix partition
  (int level, int col0, int col1,
   Context ctx, HighLevelRuntime *runtime);
  */
  
  // for node solve
  void two_level_partition(Context, HighLevelRuntime*);
  
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
  void display
  (const std::string&, Context, HighLevelRuntime*,
   bool wait=WAIT_DEFAULT);

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
  (char, char, double, const LMatrix&, const LMatrix&,
   double, LMatrix&, Context, HighLevelRuntime*,
   bool wait=WAIT_DEFAULT);
  
private:

  // ******************
  // helper functions
  // ******************

  // for init_matrix task
  ArgumentMap MapSeed(const Matrix& matrix);
  ArgumentMap MapSeed
  (const Matrix& U, const Matrix& V, const Vector& D);
  
  // to be removed 
  ArgumentMap MapSeed(int nPart, const Matrix& matrix);
  ArgumentMap MapSeed(int nPart, const Matrix& U, const Matrix& V, const Vector& D);

  // partition the matrix along rows
  IndexPartition UniformRowPartition
  (Context ctx, HighLevelRuntime *runtime);
  
  IndexPartition UniformRowPartition
  (int num_subregions, int, int, Context ctx, HighLevelRuntime *runtime);

  /*
  template <typename T>
  void solve
  (LMatrix& b, Context ctx, HighLevelRuntime *runtime,
   bool wait=WAIT_DEFAULT);
  */
  
  // ******************
  // private variables
  // ******************
  
  // matrix and block size
  int mRows;
  int mCols;
  int colIdx; // starting column index in the region
  
  // number of ranks
  // used to init data
  int              nProc; // to be removed

  // second level partition, if it exists
  int              nPart;
  int              rblock;
  int              plevel;

  // first level partition
  IndexPartition   ipart;
  LogicalPartition lpart;
  Domain           colDom;
  
  // region
  IndexSpace       ispace;
  FieldSpace       fspace;
  LogicalRegion    region;
  LogicalRegion    pregion; // to be removed
};

#endif
