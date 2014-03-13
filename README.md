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
============

template < typename dcsrmv_operator_t >
int ds_solver_mkl ( int n, int nev, int ncv, double * values, double * vectors, int ldv, dcsrmv_operator_t const & dcsrmv_operator );

template < typename dcsrmv_operator_t >
int ds_solver_gpu ( int n, int nev, int ncv, double * values, double * vectors, int ldv, dcsrmv_operator_t const & dcsrmv_operator, int dev = 0 );


CSRMV Oprators
==============

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
================

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


Utilities
=========

SAFE_CALL(call)

inline
void error_exit ( char const * fmt, ... );

inline
void warn ( char const * fmt, ... );


Deprecated
==========

template < typename dcsrmv_operator_t >
int ds_solver_gpu_blocked ( int n, int nev, int ncv, double * values, double * vectors, int ldv, dcsrmv_operator_t const & dcsrmv_operator, int dev = 0 );

class dcsrmv_operator_naive {
public:
        dcsrmv_operator_naive ( class dcsr_matrix_t const & m ) : m ( m ) { }
        ~dcsrmv_operator_naive ( ) { }
        void operator () ( double const * x, double * y ) const {
}

class dcsrmv_operator_gpu_blocked {
public:
        dcsrmv_operator_gpu_blocked ( class dcsr_matrix_t const & m, int dev = 0, size_t block_size = 0 );
        ~dcsrmv_operator_gpu_blocked ( );
        void operator () ( double const * x, double * y ) const;
}
