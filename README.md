HLanc: Heterogeneous Implicitly Restarted Lanczos Method
========================================================

HLanc is an implementation of the Implicitly restarted Lanczos method(IRLM),
and is used for solving the partial eigenvalue problems of large symmetric
sparse matrices.

The HLanc library is designed with separated heterogeneous parallel
IRLM solvers and sparse matrix-vector multiplication (SPMV) operators.
The SPMV operators hide the details about the storage of sparse matrices
from the IRLM solvers, so the solvers can work with any spare matrix format.


IRLM Solvers
------------

The IRLM solvers are implementations of the IRLM algorithm on different hardware platforms. The IRLM solvers execute the sparse matrix-vector multiplication by calling the SPMV operators, and thus contain only dense matrix operations in the algorithm. There are two solvers for double precision symmetric sparse matrix implemented in the HLanc library:

    ds_solver_mkl
    ds_solver_gpu

The ds_solver_mkl calls the BLAS and LAPACK routines in Intel MKL to perform the operations in the algorithm. The ds_solver_gpu calls CUDA kernel functions, CUDA API and cuBLAS to some operations. 

    template < typename dcsrmv_operator_t >
    int ds_solver_mkl ( int n, int nev, int ncv, double * values, double * vectors, int ldv,
                        dcsrmv_operator_t const & dcsrmv_operator );

    template < typename dcsrmv_operator_t >
    int ds_solver_gpu ( int n, int nev, int ncv, double * values, double * vectors, int ldv,
                        dcsrmv_operator_t const & dcsrmv_operator, int dev = 0 );

Arguments:

                 n: scale of the symmetric sparse matrix A.
               nev: number of eigenvalues and eigenvectors wanted.
               ncv: size of the eigenspace, usually is set to 3 times of nev.
            values: host array of length ncv, in which eigenvalues are stored in return.
           vectors: host array of size ldv * ncv, in which eigenvectors are stored in return (in column major).
               ldv: the leading dimension of the array vectors, ldv >= n.
    dspmv_operator: do the SPMV operation y = A * x.
               dev: which GPU device to use.


CSRMV Oprators
--------------

Several SPMV operators for the double precision sparse matrix with CSR format are designed in the HLanc library. As indicated by the name, dcsrmv_operator_mkl calls mkl_dcsrmv() to do SPMV on CPU, dcsrmv_operator_gpu calls cusparseDcsrmv() to do SPMV on GPU, and dcsrmv_operator_mgpu calls cusparseDcsrmv() to do SPMV on multiple GPUs.

    class dcsrmv_operator_mkl {
    public:
            dcsrmv_operator_mkl ( class dcsr_matrix_t const & m );
            ~dcsrmv_operator_mkl ( );
            void operator () ( double const * x, double * y ) const;
    };

    class dcsrmv_operator_gpu {
    public:
            dcsrmv_operator_gpu ( class dcsr_matrix_t const & m, int dev = 0 );
            ~dcsrmv_operator_gpu ( );
            void operator () ( double const * x, double * y ) const;
    };

    class dcsrmv_operator_mgpu {
    public:
            dcsrmv_operator_mgpu ( class dcsr_matrix_t const & m, unsigned mask = ( unsigned ) -1 );
            ~dcsrmv_operator_mgpu ( );
            void operator () ( double const * x, double * y ) const;
    };


CSR Matrix Class
----------------

dcsr_matrix_t reads a sparse matrix file, converts the matrix into the CSR format and saves it in the host memory.

    class dcsr_matrix_t {
    public:
            dcsr_matrix_t ( char const * format, char const * filename, bool is_symmetric = false );
            ~dcsr_matrix_t ( );
    
            void save_pp  ( char const * filename ) const;
            void save_mm  ( char const * filename ) const;
            void save_bin ( char const * filename ) const;
    
            int            get_num_rows     ( ) const;
            int            get_nnz          ( ) const;
            int            get_num_columns  ( ) const;
            int const    * get_row_ptr      ( ) const;
            int const    * get_column_index ( ) const;
            double const * get_value        ( ) const;
            bool           symmetric        ( ) const;
    };

Arguments:

          format: "mm", "pp", or "bin". "pp" for "plain presentation".
        filename: the name of the sparse matrix file.
    is_symmetric: only pp format needs this argument to indicate whether the file omits the upper half of the sparse matrix.


Utilities
---------

    SAFE_CALL(call)
    inline void error_exit ( char const * fmt, ... );
    inline void warn ( char const * fmt, ... );


Deprecated
----------

Interfaces and classes listed here are deprecated. They are with performance problem of are for debug purpose.

    template < typename dcsrmv_operator_t >
    int ds_solver_gpu_blocked ( int n, int nev, int ncv, double * values, double * vectors, int ldv,
                                dcsrmv_operator_t const & dcsrmv_operator, int dev = 0 );
    
    class dcsrmv_operator_naive {
    public:
            dcsrmv_operator_naive ( class dcsr_matrix_t const & m );
            ~dcsrmv_operator_naive ( );
            void operator () ( double const * x, double * y ) const;
    };

    class dcsrmv_operator_gpu_blocked {
    public:
            dcsrmv_operator_gpu_blocked ( class dcsr_matrix_t const & m, int dev = 0, size_t block_size = 0 );
            ~dcsrmv_operator_gpu_blocked ( );
            void operator () ( double const * x, double * y ) const;
    };
