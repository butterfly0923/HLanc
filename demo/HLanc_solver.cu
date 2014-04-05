#include <cstdio>
#include <sys/time.h>

#include "../HLanc.hpp"

struct dcsrmv_wrapper_t {
	dcsrmv_wrapper_t ( HLanc::dcsrmv_operator_gpu const & op ) : op ( op ), seq ( 0 ) { }
	void operator () ( double const * x, double * y ) const {
		fprintf ( stderr, "dcsrmv operation for the %d time ...\n", ++seq );
		op ( x, y );
	}
	HLanc::dcsrmv_operator_gpu const & op;
	mutable int seq;
};

void print_matrix_info ( HLanc::dcsr_matrix_t const & dcsr_matrix ) {
	HLanc::warn ( "%10s: %6d\n",    "rows", dcsr_matrix.get_num_rows    ( ) );
	HLanc::warn ( "%10s: %6d\n", "columns", dcsr_matrix.get_num_columns ( ) );
	HLanc::warn ( "%10s: %6d\n",     "nnz", dcsr_matrix.get_nnz         ( ) );
}

int main ( int argc, char * argv[] ) {
	if ( 5 > argc || 7 < argc ) {
		HLanc::warn ( "Usage: %s <bin|mm|pp> <matrix_file> <nev> <ncv> [dev_csrmv=0] [dev_solver=0]\n", argv[0] );
		return 1;
	}

	char const * fmt        = argv[1];
	char const * mat_file   = argv[2];
	int          nev        = atoi ( argv[3] );
	int          ncv        = atoi ( argv[4] );
	int          dev_csrmv  = 0;
	int          dev_solver = 0;
	if ( 5 < argc ) { dev_csrmv  = atoi ( argv[5] ); }
	if ( 6 < argc ) { dev_solver = atoi ( argv[6] ); }

	HLanc::warn ( "Reading a symmetric matrix from %s file \"%s\" ...\n", fmt, mat_file );
	HLanc::dcsr_matrix_t dcsr_matrix ( fmt, mat_file );
	print_matrix_info ( dcsr_matrix );
	HLanc::warn ( "Constructing a GPU CSRMV operator ...\n" );
	HLanc::dcsrmv_operator_gpu op ( dcsr_matrix, dev_csrmv );
	HLanc::warn ( "Computing the first %d eigenvalue and eigenvectors ...\n" );

	int n = dcsr_matrix.get_num_rows ( );
	double * values  = new double[ncv];
	double * vectors = new double[n * ncv];

	struct timeval tb, te;
	gettimeofday ( &tb, NULL );
	int ret = HLanc::ds_solver_gpu ( n, nev, ncv, values, vectors, n, dcsrmv_wrapper_t ( op ), dev_solver );
	gettimeofday ( &te, NULL );

	HLanc::warn ( "[31;1m%s[0m\n", 0 == ret ? "Success." : "Failes." );
	HLanc::warn ( "Time usage for the solver: %.6lf\n", te.tv_sec - tb.tv_sec + 1E-6 * ( te.tv_usec - tb.tv_usec ) );

	for ( int i = 0; i < nev; ++i ) {
		printf ( "%23.15lE\n", values[i] );
	}
	for ( int i = 0; i < n; ++i ) {
		for ( int j = 0; j < nev; ++j ) {
			printf ( "%23.15lE%c", vectors[i + j * n], j + 1 == nev ? '\n' : ' ' );
		}
	}

	delete [] values;
	delete [] vectors;

	return ret;
}
