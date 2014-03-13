#include <cstdio>

#include "../HLanc.hpp"

char const *  mm_file = "../data/Journals.mtx";
char const *  pp_file = "../data/simple.pp";
char const * bin_file = "simple.bin";

void op_fun_simple ( double const * x, double * y ) {
#define N_SIMPLE	16
	for ( int i = 0; i < N_SIMPLE; ++i ) {
		double s = 0.0;
		for ( int j = 0; j < N_SIMPLE; ++j ) {
			s += ( ( i < j ? i : j ) + 1 ) * x[j];
		}
		y[i] = s;
	}
}

void print_matrix_info ( HLanc::dcsr_matrix_t const & dcsr_matrix ) {
	HLanc::warn ( "%10s: %6d\n",    "rows", dcsr_matrix.get_num_rows    ( ) );
	HLanc::warn ( "%10s: %6d\n", "columns", dcsr_matrix.get_num_columns ( ) );
	HLanc::warn ( "%10s: %6d\n",     "nnz", dcsr_matrix.get_nnz         ( ) );
}

int main ( int argc, char * argv[] ) {
	HLanc::warn ( "=========================================\n" );
	HLanc::warn ( "This is a demostration program for HLanc.\n" );
	HLanc::warn ( "=========================================\n" );
	HLanc::warn ( "\n" );
	HLanc::warn ( "HLanc is an implementation of the Implicitly restarted Lanczos method(IRLM),\n" );
	HLanc::warn ( "  and is used for solving the partial eigenvalue problems of large symmetric\n" );
	HLanc::warn ( "  sparse matrices.\n" );
	HLanc::warn ( "\n" );
	HLanc::warn ( "The HLanc library is designed with separated heterogeneous parallel\n" );
	HLanc::warn ( "  IRLM solvers and sparse matrix-vector multiplication (SPMV) operators.\n" );
	HLanc::warn ( "  The SPMV operators hide the details about the storage of sparse matrices\n" );
	HLanc::warn ( "  from the IRLM solvers, so the solvers can work with any spare matrix format.\n" );
	HLanc::warn ( "\n" );
	HLanc::warn ( "Following are sample use cases.\n" );
	HLanc::warn ( "\n" );

	int n, nev, ncv, ret;
	double * values;
	double * vectors;

	// MatrixMarket file
	HLanc::warn ( "Reading a symmetric matrix from a MatrixMarket file \"%s\" ...\n", mm_file );
	HLanc::dcsr_matrix_t dcsr_matrix_mm ( "mm", mm_file );
	print_matrix_info ( dcsr_matrix_mm );
	HLanc::warn ( "Constructing a GPU CSRMV operator ...\n" );
	HLanc::dcsrmv_operator_gpu op_mm ( dcsr_matrix_mm );
	HLanc::warn ( "Computing the first 4 eigenvalue and eigenvectors with CPU+GPU ...\n" );
	n = dcsr_matrix_mm.get_num_rows ( );
	nev = 4;
	ncv = nev * 3;
	values = new double[ncv];
	vectors = new double[n * ncv];
	ret = HLanc::ds_solver_gpu ( n, nev, ncv, values, vectors, n, op_mm );
	HLanc::warn ( "%s\n", 0 == ret ? "Success." : "Failes." );
	HLanc::warn ( "The eigenvalues are:\n" );
	for ( int i = 0; i < nev; ++i ) {
		HLanc::warn ( "%23.15lE\n", values[i] );
	}
	HLanc::warn ( "Computing the first 4 eigenvalue and eigenvectors with MKL ...\n" );
	ret = HLanc::ds_solver_mkl ( n, nev, ncv, values, vectors, n, op_mm );
	HLanc::warn ( "%s\n", 0 == ret ? "Success." : "Failes." );
	HLanc::warn ( "You should see results close to that above but with float point errors.\n" );
	HLanc::warn ( "The eigenvalues are:\n" );
	for ( int i = 0; i < nev; ++i ) {
		HLanc::warn ( "%23.15lE\n", values[i] );
	}
	HLanc::warn ( "The eigenvectors are stored in column major order.\n" );
	HLanc::warn ( "In this case we omit the output of the eigenvectors as they are a bit long.\n" );
	HLanc::warn ( "\n" );
	delete [] values;
	delete [] vectors;

	// plain file
	HLanc::warn ( "Reading a symmetric matrix from a plain file \"%s\" ...\n", pp_file );
	HLanc::dcsr_matrix_t dcsr_matrix_pp ( "pp", pp_file );
	print_matrix_info ( dcsr_matrix_pp );
	HLanc::warn ( "Constructing a GPU CSRMV operator ...\n" );
	HLanc::dcsrmv_operator_gpu op_pp ( dcsr_matrix_pp );
	HLanc::warn ( "Computing the first 4 eigenvalue and eigenvectors with CPU+GPU ...\n" );
	n = dcsr_matrix_pp.get_num_rows ( );
	nev = 4;
	ncv = nev * 3;
	values = new double[ncv];
	vectors = new double[n * ncv];
	ret = HLanc::ds_solver_gpu ( n, nev, ncv, values, vectors, n, op_pp );
	HLanc::warn ( "%s\n", 0 == ret ? "Success." : "Failes." );
	HLanc::warn ( "The eigenvalues are:\n" );
	for ( int i = 0; i < nev; ++i ) {
		HLanc::warn ( "%23.15lE\n", values[i] );
	}
	HLanc::warn ( "The eigenvectors are stored in column major order.\n" );
	for ( int i = 0; i < n; ++i ) {
		for ( int j = 0; j < nev; ++j ) {
			HLanc::warn ( "%23.15lE%c", vectors[i + j * n], j + 1 == nev ? '\n' : ' ' );
		}
	}
	HLanc::warn ( "\n" );
	HLanc::warn ( "Computing the first 4 eigenvalue and eigenvectors with MKL ...\n" );
	ret = HLanc::ds_solver_mkl ( n, nev, ncv, values, vectors, n, op_pp );
	HLanc::warn ( "%s\n", 0 == ret ? "Success." : "Failes." );
	HLanc::warn ( "The eigenvalues are:\n" );
	for ( int i = 0; i < nev; ++i ) {
		HLanc::warn ( "%23.15lE\n", values[i] );
	}
	HLanc::warn ( "The eigenvectors are stored in column major order.\n" );
	for ( int i = 0; i < n; ++i ) {
		for ( int j = 0; j < nev; ++j ) {
			HLanc::warn ( "%23.15lE%c", vectors[i + j * n], j + 1 == nev ? '\n' : ' ' );
		}
	}
	HLanc::warn ( "\n" );
	delete [] values;
	delete [] vectors;

	// binary file
	HLanc::warn ( "Converting the plain file to binary format ...\n" );
	dcsr_matrix_pp.save_bin ( bin_file );
	HLanc::warn ( "Reading a symmetric matrix from the binary file \"%s\" ...\n", bin_file );
	HLanc::dcsr_matrix_t dcsr_matrix_bin ( "bin", bin_file );
	print_matrix_info ( dcsr_matrix_bin );
	HLanc::warn ( "Constructing a GPU CSRMV operator ...\n" );
	HLanc::dcsrmv_operator_gpu op_bin ( dcsr_matrix_bin );
	HLanc::warn ( "Computing the first 4 eigenvalue and eigenvectors with CPU+GPU ...\n" );
	n = dcsr_matrix_bin.get_num_rows ( );
	nev = 4;
	ncv = nev * 3;
	values = new double[ncv];
	vectors = new double[n * ncv];
	ret = HLanc::ds_solver_gpu ( n, nev, ncv, values, vectors, n, op_bin );
	HLanc::warn ( "%s\n", 0 == ret ? "Success." : "Failes." );
	HLanc::warn ( "You should see the same results.\n" );
	HLanc::warn ( "The eigenvalues are:\n" );
	for ( int i = 0; i < nev; ++i ) {
		HLanc::warn ( "%23.15lE\n", values[i] );
	}
	HLanc::warn ( "The eigenvectors are stored in column major order.\n" );
	for ( int i = 0; i < n; ++i ) {
		for ( int j = 0; j < nev; ++j ) {
			HLanc::warn ( "%23.15lE%c", vectors[i + j * n], j + 1 == nev ? '\n' : ' ' );
		}
	}
	HLanc::warn ( "\n" );
	delete [] values;
	delete [] vectors;

	// using function as SPMV operator
	HLanc::warn ( "The IRLM solvers are able to recieve SPMV operators in any type.\n" );
	HLanc::warn ( "  In this case, the results of the matrix in the simple case is\n" );
	HLanc::warn ( "  computed with a function used as an SPMV operator. The matrix\n" );
	HLanc::warn ( "  is not stored explicitly.\n" );
	HLanc::warn ( "\n" );
	n = N_SIMPLE;
	nev = 4;
	ncv = nev * 3;
	values = new double[ncv];
	vectors = new double[n * ncv];
	HLanc::warn ( "Computing the first 4 eigenvalue and eigenvectors with CPU+GPU ...\n" );
	ret = HLanc::ds_solver_gpu ( n, nev, ncv, values, vectors, n, op_fun_simple );
	HLanc::warn ( "%s\n", 0 == ret ? "Success." : "Failes." );
	HLanc::warn ( "You should also see the same results.\n" );
	HLanc::warn ( "The eigenvalues are:\n" );
	for ( int i = 0; i < nev; ++i ) {
		HLanc::warn ( "%23.15lE\n", values[i] );
	}
	HLanc::warn ( "The eigenvectors are stored in column major order.\n" );
	for ( int i = 0; i < n; ++i ) {
		for ( int j = 0; j < nev; ++j ) {
			HLanc::warn ( "%23.15lE%c", vectors[i + j * n], j + 1 == nev ? '\n' : ' ' );
		}
	}
	HLanc::warn ( "\n" );
	delete [] values;
	delete [] vectors;

	return 0;
}
