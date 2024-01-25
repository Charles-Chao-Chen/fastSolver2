#ifndef _lmatrix_hpp
#define _lmatrix_hpp

#include <string>

#include "legion.h"
using namespace Legion;

#include "utility.hpp" // for WAIT_DEFAULT and FIELDID_V
#include "matrix.hpp"
#include "solver_tasks.hpp"

// legion matrix
class LMatrix {
public:
  LMatrix();
  LMatrix(int, int, int, Context, Runtime*);
  LMatrix(int, int, LogicalRegion, IndexSpace, FieldSpace);
  LMatrix(LogicalRegion r, int rows, int cols);

  /*
  LMatrix
  (int, int, int, IndexPartition ip, LogicalRegion lr,
   Context, Runtime*); // to be removed
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
  int small_block_parts() const;
  
  void set_column_size(int);
  void set_column_begin(int);
  void set_logical_region(LogicalRegion);
  void set_parent_region(LogicalRegion);
  void set_logical_partition(LogicalPartition lp);
  
  // create logical region
  void create(int, int, Context, Runtime*);

  // set the matrix to value
  void clear
  (double, Context, Runtime*, bool wait=WAIT_DEFAULT);

  // scale all the entries
  void scale
  (double, Context, Runtime*, bool wait=WAIT_DEFAULT);

  // initialize data from Matrix object
  void init_data
  (const Matrix& mat, Context, Runtime*,
   bool wait=WAIT_DEFAULT);

  // init part of the region
  void init_data
  (int, int, const Matrix& mat, Context, Runtime*,
   bool wait=WAIT_DEFAULT);
  
  // output region
  Matrix to_matrix(Context, Runtime*);
  Matrix to_matrix(int, int, Context, Runtime*);
  Matrix to_matrix(int, int, int, int, Context, Runtime*);

  // to be removed
  void init_data
  (int, const Matrix& VMat, Context, Runtime*,
   bool wait=WAIT_DEFAULT);

  // specifies column range
  // for UTree init
  void init_data
  (int, int, int, const Matrix& VMat, Context, Runtime*,
   bool wait=WAIT_DEFAULT);
  
  // for KTree::partition
  void init_dense_blocks
  (const Matrix& UMat, const Matrix& VMat, const Vector& DVec,
   Context, Runtime*, bool wait=WAIT_DEFAULT);

  void init_dense_blocks
  (int, int, const Matrix& UMat, const Matrix& VMat, const Vector& DVec,
   Context, Runtime*, bool wait=WAIT_DEFAULT);

  // uniform partition
  // Note: the first argument is level, NOT nPart
  void partition(int level, Context, Runtime*);

  /*
  // return a new matrix with created partition,
  //  so no need to destroy region twice
  LMatrix partition
  (int level, int col0, int col1,
   Context ctx, Runtime *runtime);
  */
  
  // for node solve
  void two_level_partition(Context, Runtime*);
  
  // output the right hand side
  // for UTree::rhs()
  Vector to_vector();

  // for KTree::partition()
  //void create_dense_partition
  //(int, const Matrix& U, const Matrix& V, const Vector& D,
  //Context, Runtime*, bool wait=WAIT_DEFAULT);
  
  // solve linear system
  // for KTree::solve()
  void solve
  (LMatrix&, LMatrix&, Context, Runtime*, bool wait=WAIT_DEFAULT);

  // solve node system
  // for HMatrix::solve()
  void node_solve
  (LMatrix&, Context, Runtime*, bool wait=WAIT_DEFAULT);

  static void node_solve
  (LMatrix&, LMatrix&, LMatrix&, LMatrix&,
   PhaseBarrier pb_wait, PhaseBarrier pb_ready,
   Context ctx, Runtime* runtime, bool wait=WAIT_DEFAULT);

  // print the values on screen
  // for debugging
  void display
  (const std::string&, Context, Runtime*,
   bool wait=WAIT_DEFAULT);

  // free resources
  void clear(Context, Runtime*);
  
  // static methods
  // matrix subtraction
  static void add
  (double alpha, const LMatrix&,
   double beta,  const LMatrix&, LMatrix&,
   Context, Runtime*, bool wait=WAIT_DEFAULT);

  // gemm reduction
  // C = alpha*op(A) * op(B) + beta*C
  static void gemmRed
  (char, char, double, const LMatrix&, const LMatrix&,
   double, LMatrix&, Context, Runtime*,
   bool wait=WAIT_DEFAULT);

  static void gemm
  (char, char, double, const LMatrix&, const LMatrix&,
   double, LMatrix&, Context, Runtime*,
   bool wait=WAIT_DEFAULT);

  static void gemm_inplace
  (char, char, double, const LMatrix&, const LMatrix&,
   double, LMatrix&, Context, Runtime*,
   bool wait=WAIT_DEFAULT);

  // gemm broadcast
  static void gemmBro
  (char, char, double, const LMatrix&, const LMatrix&,
   double, LMatrix&, Context, Runtime*,
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
  (Context ctx, Runtime *runtime);
  
  IndexPartition UniformRowPartition
  (int num_subregions, int, int, Context ctx, Runtime *runtime);

  /*
  template <typename T>
  void solve
  (LMatrix& b, Context ctx, Runtime *runtime,
   bool wait=WAIT_DEFAULT);
  */
  
  // ******************
  // private variables
  // ******************
  
  // matrix and block size
  int mRows;
  int mCols;
  int colIdx;   // starting column index in the region
  int smallblk; // number of blocks in every partition,
                //  used when treelvel != launchlvl
  
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
