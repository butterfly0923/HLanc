#include <cstdio>
#include <cstdlib>

#include "../HLanc.hpp"

struct eigs_t {
	int nev;
	int n;
	double * value;
	double * vector;
	eigs_t ( char const * filename ) {
		FILE * fp = fopen ( filename, "r" );
		if ( !fp ) {
			HLanc::warn ( "Cannot open file \"%s\": %m\n", filename );
			exit ( 1 );
		}

		if ( 1 != fscanf ( fp, "%d", &nev ) ) {
			HLanc::warn ( "Error reading nev from \"%s\".\n", filename );
			exit ( 1 );
		}
		if ( 1 != fscanf ( fp, "%d", &n ) ) {
			HLanc::warn ( "Error reading n from \"%s\".\n", filename );
			exit ( 1 );
		}

		value  = new double[nev];
		vector = new double[n * nev];
		for ( int i = 0; i < nev; ++i ) {
			if ( 1 != fscanf ( fp, "%lE", &value[i] ) ) {
				HLanc::warn ( "Error reading eig value at index %d from \"%s\".\n", i, filename );
				exit ( 1 );
			}
		}
		for ( int r = 0; r < n; ++r ) {
			for ( int c = 0; c < nev; ++c ) {
				if ( 1 != fscanf ( fp, "%lE", &vector[c * n + r] ) ) {
					HLanc::warn ( "Error reading eig vector at index <%d,%d> from \"%s\".\n", r, c, filename );
					exit ( 1 );
				}
			}
		}

		fclose ( fp );
	}
	~eigs_t ( ) {
		delete [] value;
		delete [] vector;
	}
};

double rdiff ( double const * x, double const * y, int len ) {
	double maxdiff = 0;
	double maxabs = 0;
	for ( int i = 0; i < len; ++i ) {
		double d1 = x[i];
		double d2 = y[i];
		double diff = d1 - d2;
		if ( diff < 0 ) diff = -diff;
		maxabs = max ( maxabs, abs ( d1 ) );
		maxabs = max ( maxabs, abs ( d2 ) );
		if ( maxdiff < diff ) maxdiff = diff;
	}
	if ( maxabs ) maxdiff /= maxabs;
	return maxdiff;
}

int main ( int argc, char * argv[] ) {
	if ( 4 != argc && 5 != argc ) {
		HLanc::warn ( "Usage: %s <bin|mm|pp> <matrix_file> <eigs_file> [dev=0]\n", argv[0] );
		return 1;
	}

	char const *   format = argv[1];
	char const * mat_file = argv[2];
	HLanc::dcsr_matrix_t m ( format, mat_file );

	eigs_t eigs ( argv[3] );
	int dev = 0;
	if ( 4 < argc ) { dev = atoi ( argv[4] ); }

	int nev = eigs.nev;
	int n   = eigs.n;

	HLanc::dcsrmv_operator_gpu op ( m, dev );

	double * x = new double[n];
	double * y = new double[n];

	for ( int c = 0; c < nev; ++c ) {
		double   value  = eigs.value[c];
		double * vector = eigs.vector + c * n;
		op ( vector, x );
		for ( int r = 0; r < n; ++r ) {
			y[r] = vector[r] * value;
		}
		HLanc::warn ( "max r-diff of vector[%3d]: %.6lE\n", c, rdiff ( x, y, n ) );
	}

	delete [] y;
	delete [] x;
	return 0;
}
