#include <cstdio>
#include <cstdlib>

#include "../HLanc.hpp"

void print_matrix_info ( HLanc::dcsr_matrix_t const & dcsr_matrix ) {
	HLanc::warn ( "%10s: %6d\n",    "rows", dcsr_matrix.get_num_rows    ( ) );
	HLanc::warn ( "%10s: %6d\n", "columns", dcsr_matrix.get_num_columns ( ) );
	HLanc::warn ( "%10s: %6d\n",     "nnz", dcsr_matrix.get_nnz         ( ) );
}

int main ( int argc, char * argv[] ) {
	if ( 4 != argc && 5 != argc ) {
		HLanc::warn ( "matrix_convertor: convert a sparse matrix into binary csr format.\n" );
		HLanc::warn ( "Usage: %s <mm|pp> <matrix_file> <binary_file>\n", argv[0] );
		return 1;
	}

	char const *   format = argv[1];
	char const * mat_file = argv[2];
	char const * bin_file = argv[3];

	HLanc::warn ( "Reading %s matrix from %s ...\n", format, mat_file );
	HLanc::dcsr_matrix_t m ( format, mat_file );
	print_matrix_info ( m );

	HLanc::warn ( "Saving matrix into binary file %s ...\n", bin_file );
	m.save_bin ( bin_file );
	HLanc::warn ( "Success.\n" );

	return 0;
}
