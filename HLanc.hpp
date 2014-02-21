#pragma once

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <functional>
#include <mkl.h>
#include <vector>

#define HLANC_VERSION_MAJOR	0
#define HLANC_VERSION_MINOR	1

#define CEIL_DIV( a, b ) ( ( ( a ) + ( b ) - 1 ) / ( b ) )
#define MIN(a,b)         ( ( a ) < ( b ) ? ( a ) : ( b ) )
#define MAX(a,b)         ( ( a ) > ( b ) ? ( a ) : ( b ) )

double const c_23  =  0.66666666666666663;
double const c_d0  =  0.0;
double const c_d1  =  1.0;
double const c_dn1 = -1.0;


/////////////////////
// HLanc Namespace //
/////////////////////

namespace HLanc {

/****************
 ** Utils **
 ****************/

template < typename status_t > inline
void check_call ( status_t error, char const * file, size_t line, char const * call ) {
	if ( 0 == error ) return;
	fprintf ( stderr, "%s failed in \"%s:%lu\": %d\n", call, file, line, ( int ) error );
	exit ( error );
}
template < > inline
void check_call < cudaError_t > ( cudaError_t error, char const * file, size_t line, char const * call ) {
	if ( cudaSuccess == error ) return;
	fprintf ( stderr, "%s failed in \"%s:%lu\": %s\n", call, file, line, cudaGetErrorString ( error ) );
	exit ( error );
}
#define SAFE_CALL(call) check_call ( ( call ), __FILE__, __LINE__, #call )

inline
void error_exit ( char const * fmt, ... ) {
	va_list ap;
	va_start ( ap, fmt );
	vfprintf ( stderr, fmt, ap );
	va_end ( ap );
	exit ( 1 );
}

inline
void warn ( char const * fmt, ... ) {
	va_list ap;
	va_start ( ap, fmt );
	vfprintf ( stderr, fmt, ap );
	va_end ( ap );
}


/*******************
 ** dcsr_matrix_t **
 *******************/

class dcsr_matrix_t {
public:
	dcsr_matrix_t ( char const * format, char const * filename, bool is_symmetric = false );
	~dcsr_matrix_t ( );

	void save_pp  ( char const * filename ) const;
	void save_mm  ( char const * filename ) const;
	void save_bin ( char const * filename ) const;

	int            get_num_rows     ( ) const { return csr_row_ptr.size ( ) - 1;            }
	int            get_nnz          ( ) const { return csr_value.size ( );                  }
	int            get_num_columns  ( ) const { return num_columns;                         }
	int const    * get_row_ptr      ( ) const { return csr_row_ptr.begin ( ).base ( );      }
	int const    * get_column_index ( ) const { return csr_column_index.begin ( ).base ( ); }
	double const * get_value        ( ) const { return csr_value.begin ( ).base ( );        }
	bool           symmetric        ( ) const { return is_symmetric;                        }
protected:
	void read_pp  ( char const * filename, bool is_symmetric = false );
	void read_mm  ( char const * filename );
	void read_bin ( char const * filename );

	std::vector <    int > csr_row_ptr;
	std::vector <    int > csr_column_index;
	std::vector < double > csr_value;
	int num_columns;
	bool is_symmetric;
};

inline
dcsr_matrix_t::dcsr_matrix_t ( char const * format, char const * filename, bool is_symmetric ) {
	if ( !strcmp ( format, "pp" ) ) {
		read_pp ( filename, is_symmetric );
	} else if ( !strcmp ( format, "mm" ) ) {
		read_mm ( filename );
	} else if ( !strcmp ( format, "bin" ) ) {
		read_bin ( filename );
	} else {
		error_exit ( "Unsupported format: %s\n", format );
	}
}

inline
dcsr_matrix_t::~dcsr_matrix_t ( ) {
}

inline
void dcsr_matrix_t::read_pp ( char const * filename, bool is_symmetric ) {
	FILE * fp;
	if ( !( fp = fopen ( filename, "r" ) ) ) {
		error_exit ( "Error opening file %s: %m\n", filename );
	}

	csr_row_ptr     .clear ( );
	csr_column_index.clear ( );
	csr_value       .clear ( );
	this->is_symmetric = is_symmetric;

	for ( num_columns = 0, csr_row_ptr.push_back ( 0 ); ; ) {
		int ich;
		while ( ich = fgetc ( fp ), isblank ( ich ) ) { }

		if ( '\n' == ich ) {
			csr_row_ptr.push_back ( csr_value.size ( ) );
			continue;
		}

		if ( EOF == ich ) {
			break;
		}

		ungetc ( ich, fp );
		int index;
		double value;
		if ( 2 != fscanf ( fp, "%d:%lf", &index, &value ) ) {
			error_exit ( "Error reading file %s at line %d.\n", filename, csr_row_ptr.size ( ) );
		}

		if ( 0.0 == value ) {
			continue;
		}

		int current_row = csr_row_ptr.size ( ) - 1;
		if ( is_symmetric && index > current_row ) {
			warn ( "Ignore element [%d, %d] = %lg\n", current_row, index, value );
		} else {
			csr_column_index.push_back ( index );
			csr_value.push_back ( value );
		}

		if ( num_columns < index ) num_columns = index;
		if ( is_symmetric && num_columns < current_row ) num_columns = current_row;
	}
	++num_columns;

	fclose ( fp );
}

inline
void dcsr_matrix_t::read_mm ( char const * filename ) {
	FILE * fp;
	if ( !( fp = fopen ( filename, "r" ) ) ) {
		error_exit ( "Error opening file %s: %m\n", filename );
	}

	csr_row_ptr     .clear ( );
	csr_column_index.clear ( );
	csr_value       .clear ( );

	char * line = NULL;
	size_t len = 0;
	if ( -1 == getline ( &line, &len, fp ) ) {
		error_exit ( "Error reading banner in file %s\n", filename );
	}

	if ( line != strstr ( line, "%%MatrixMarket matrix coordinate" ) ) {
		error_exit ( "Unknown banner format: %s\n", line );
	}

	if ( strstr ( line, "symmetric" ) ) {
		is_symmetric = true;
	} else if ( strstr ( line, "general" ) ) {
		is_symmetric = false;
	} else {
		error_exit ( "Unknown banner format: %s\n", line );
	}

	int ich;
	while ( '%' == ( ich = fgetc ( fp ) ) ) {
		while ( '\n' != fgetc ( fp ) ) { }
	}
	ungetc ( ich, fp );

	int rows, columns, entries;
	if ( 3 != fscanf ( fp, "%d%d%d", &rows, &columns, &entries ) ) {
		error_exit ( "Error reading rows, columns and entries\n" );
	}
	if ( num_columns < columns ) num_columns = columns;

	struct line_t {
		std::vector < int > c;
		std::vector < double > v;
	} * m = new line_t[rows];

	int r, c;
	double v;
	while ( entries-- ) {
		if ( 3 != fscanf ( fp, "%d%d%lf", &r, &c, &v ) ) {
			error_exit ( "Error reading element\n" );
		}
		if ( 0.0 == v ) {
			continue;
		}
		--r, --c;
		m[r].c.push_back ( c );
		m[r].v.push_back ( v );
	}
	fclose ( fp );

	for ( r = 0, csr_row_ptr.push_back ( 0 ); r < rows; ++r ) {
		std::vector <    int >::const_iterator itc;
		std::vector < double >::const_iterator itv;
		for ( itc = m[r].c.begin ( ), itv = m[r].v.begin ( ); itc != m[r].c.end ( ); ++itc, ++itv ) {
			if ( is_symmetric && *itc > r ) {
				warn ( "Ignore element [%d, %d] = %lg\n", r, *itc, *itv );
				continue;
			}
			csr_column_index.push_back ( *itc );
			csr_value       .push_back ( *itv );
		}
		csr_row_ptr.push_back ( csr_value.size ( ) );
	}

	delete [] m;
}

inline
void dcsr_matrix_t::read_bin ( char const * filename ) {
	FILE * fp;
	if ( !( fp = fopen ( filename, "r" ) ) ) {
		error_exit ( "Error opening file %s: %m\n", filename );
	}

	is_symmetric = false;

	if ( 1 != fread ( &num_columns, sizeof ( int ), 1, fp ) ) {
		error_exit ( "Error reading num_columns.\n" );
	}

	int num_rows;
	if ( 1 != fread ( &num_rows, sizeof ( int ), 1, fp ) ) {
		error_exit ( "Error reading num_columns.\n" );
	}
	csr_row_ptr.resize ( num_rows + 1 );
	if ( num_rows + 1 != fread ( &csr_row_ptr[0], sizeof ( int ), num_rows + 1, fp ) ) {
		error_exit ( "Error reading csr_row_ptr[].\n" );
	}

	int nnz;
	if ( 1 != fread ( &nnz, sizeof ( int ), 1, fp ) ) {
		error_exit ( "Error reading nnz.\n" );
	}
	csr_column_index.resize ( nnz );
	if ( nnz != fread ( &csr_column_index[0], sizeof ( int ), nnz, fp ) ) {
		error_exit ( "Error reading csr_column_index[].\n" );
	}
	csr_value.resize ( nnz );
	if ( nnz != fread ( &csr_value[0], sizeof ( double ), nnz, fp ) ) {
		error_exit ( "Error reading csr_value[].\n" );
	}

	fclose ( fp );
}

inline
void dcsr_matrix_t::save_pp ( char const * filename ) const {
	FILE * fp;
	if ( !( fp = fopen ( filename, "w" ) ) ) {
		error_exit ( "Error opening file %s: %m\n", filename );
	}

	for ( int r = 0; r < csr_row_ptr.size ( ) - 1; ++r ) {
		for ( int i = csr_row_ptr[r]; i < csr_row_ptr[r + 1]; ++i ) {
			if ( 0 > fprintf ( fp, "%d:%lg%c", csr_column_index[i], csr_value[i], i + 1 == csr_row_ptr[r + 1] ? '\n' : ' ' ) ) {
				error_exit ( "Error writing file %s\n", filename );
			}
		}
	}

	fclose ( fp );
}

inline
void dcsr_matrix_t::save_mm ( char const * filename ) const {
	FILE * fp;
	if ( !( fp = fopen ( filename, "w" ) ) ) {
		error_exit ( "Error opening file %s: %m\n", filename );
	}

	if ( 0 > fprintf ( fp, "%%%%MatrixMarket matrix coordinate real %s\n", is_symmetric ? "symmetric" : "general" ) ) {
		error_exit ( "Error writing banner to file %s\n", filename );
	}
	if ( 0 > fprintf ( fp, "%d %d %d\n", ( int ) csr_row_ptr.size ( ) - 1, num_columns, ( int ) csr_value.size ( ) ) ) {
		error_exit ( "Error writing file %s\n", filename );
	}

	for ( int r = 0; r < csr_row_ptr.size ( ) - 1; ++r ) {
		for ( int i = csr_row_ptr[r]; i < csr_row_ptr[r + 1]; ++i ) {
			if ( 0 > fprintf ( fp, "%d %d %lg\n", r + 1, csr_column_index[i] + 1, csr_value[i] ) ) {
				error_exit ( "Error writing file %s\n", filename );
			}
		}
	}

	fclose ( fp );
}

inline
void dcsr_matrix_t::save_bin ( char const * filename ) const {
	int num_rows = csr_row_ptr.size ( ) - 1;

	struct line_t {
		std::vector < int > c;
		std::vector < double > v;
	} * m = new line_t[num_rows];

	for ( int r = 0; r < num_rows; ++r ) {
		for ( int i = csr_row_ptr[r]; i < csr_row_ptr[r + 1]; ++i ) {
			int c = csr_column_index[i];
			double v = csr_value[i];
			m[r].c.push_back ( c );
			m[r].v.push_back ( v );
			if ( is_symmetric && r != c ) {
				m[c].c.push_back ( r );
				m[c].v.push_back ( v );
			}
		}
	}

	std::vector < int > _row_ptr;
	std::vector < int > _column_index;
	std::vector < double > _value;
	_row_ptr.push_back ( 0 );
	for ( int r = 0; r < num_rows; ++r ) {
		std::vector <    int >::const_iterator itc;
		std::vector < double >::const_iterator itv;
		for ( itc = m[r].c.begin ( ), itv = m[r].v.begin ( ); itc != m[r].c.end ( ); ++itc, ++itv ) {
			_column_index.push_back ( *itc );
			_value       .push_back ( *itv );
		}
		_row_ptr.push_back ( _value.size ( ) );
	}

	FILE * fp;
	if ( !( fp = fopen ( filename, "w" ) ) ) {
		error_exit ( "Error opening file %s: %m\n", filename );
	}
	if ( 1 != fwrite ( &num_columns, sizeof ( int ), 1, fp ) ) {
		error_exit ( "Error writing num_columns.\n" );
	}
	if ( 1 != fwrite ( &num_rows, sizeof ( int ), 1, fp ) ) {
		error_exit ( "Error writing num_rows.\n" );
	}
	if ( num_rows + 1 != fwrite ( &_row_ptr[0], sizeof ( int ), num_rows + 1, fp ) ) {
		error_exit ( "Error writing row_ptr[].\n" );
	}
	int nnz = _column_index.size ( );
	if ( 1 != fwrite ( &nnz, sizeof ( int ), 1, fp ) ) {
		error_exit ( "Error writing nnz.\n" );
	}
	if ( nnz != fwrite ( &_column_index[0], sizeof ( int ), nnz, fp ) ) {
		error_exit ( "Error writing row_ptr[].\n" );
	}
	if ( nnz != fwrite ( &_value[0], sizeof ( double ), nnz, fp ) ) {
		error_exit ( "Error writing value[].\n" );
	}
	fclose ( fp );

	delete [] m;
}


/***************************
 ** dcsrmv_operator_naive **
 ***************************/

class dcsrmv_operator_naive {
public:
	dcsrmv_operator_naive ( class dcsr_matrix_t const & m ) : m ( m ) { }
	~dcsrmv_operator_naive ( ) { }
	void operator () ( double const * x, double * y ) const {
		int num_rows = m.get_num_rows ( );
		int const * row_ptr = m.get_row_ptr ( );
		int const * column_index = m.get_column_index ( );
		double const * values = m.get_value ( );
		if ( m.symmetric ( ) ) {
			for ( int i = 0; i < num_rows; y[i++] = 0.0 ) { }
			for ( int r = 0; r < num_rows; ++r ) {
				for ( int i = row_ptr[r]; i < row_ptr[r + 1]; ++i ) {
					int c = column_index[i];
					double v = values[i];
					y[r] += x[c] * v;
					if ( r != c ) {
						y[c] += x[r] * v;
					}
				}
			}
		} else {
			for ( int r = 0; r < num_rows; ++r ) {
				double s = 0.0;
				for ( int i = row_ptr[r]; i < row_ptr[r + 1]; ++i ) {
					int c = column_index[i];
					double v = values[i];
					s += x[c] * v;
				}
				y[r] = s;
			}
		}
	}
protected:
	class dcsr_matrix_t const & m;
};


/*************************
 ** dcsrmv_operator_mkl **
 *************************/

class dcsrmv_operator_mkl {
public:
	dcsrmv_operator_mkl ( class dcsr_matrix_t const & m ) : m ( m ) { }
	~dcsrmv_operator_mkl ( ) { }
	void operator () ( double const * x, double * y ) const {
		int m = this->m.get_num_rows ( );
		int k = this->m.get_num_columns ( );
		double alpha = 1.0;
		double  beta = 0.0;
		char matdescra[6] = "GLNC";
		if ( this->m.symmetric ( ) ) matdescra[0] = 'S';
		mkl_dcsrmv (
			"N",
			&m,
			&k,
			&alpha,
			matdescra,
			( double * ) this->m.get_value ( ),
			(    int * ) this->m.get_column_index ( ),
			(    int * ) this->m.get_row_ptr ( ),
			(    int * ) this->m.get_row_ptr ( ) + 1,
			( double * ) x,
			&beta,
			y
		);
	}
protected:
	class dcsr_matrix_t const & m;
};


/*************************
 ** dcsrmv_operator_gpu **
 *************************/

class dcsrmv_operator_gpu {
public:
	dcsrmv_operator_gpu ( class dcsr_matrix_t const & m, int dev = 0 );
	~dcsrmv_operator_gpu ( );
	void operator () ( double const * x, double * y ) const;
protected:
	class dcsr_matrix_t const & m;
	int dev;
	int    * csr_row_ptr_d;
	int    * csr_column_index_d;
	double * csr_value_d;
	double * x_d;
	double * y_d;
	cusparseHandle_t handle;
	cusparseMatDescr_t descr;
};

inline
dcsrmv_operator_gpu::dcsrmv_operator_gpu ( class dcsr_matrix_t const & m, int dev ) : m ( m ), dev ( dev ) {
	int num_rows    = this->m.get_num_rows    ( );
	int num_columns = this->m.get_num_columns ( );
	int nnz         = this->m.get_nnz         ( );

	SAFE_CALL ( cudaSetDevice ( dev ) );
	SAFE_CALL ( cudaMalloc ( ( void ** ) &csr_row_ptr_d,      ( num_rows + 1 ) * sizeof (    int ) ) );
	SAFE_CALL ( cudaMalloc ( ( void ** ) &csr_column_index_d,              nnz * sizeof (    int ) ) );
	SAFE_CALL ( cudaMalloc ( ( void ** ) &csr_value_d,                     nnz * sizeof ( double ) ) );
	SAFE_CALL ( cudaMalloc ( ( void ** ) &x_d,                     num_columns * sizeof ( double ) ) );
	SAFE_CALL ( cudaMalloc ( ( void ** ) &y_d,                        num_rows * sizeof ( double ) ) );
	SAFE_CALL ( cudaMemset (              y_d, 0x00,                  num_rows * sizeof ( double ) ) );
	SAFE_CALL ( cudaMemcpy ( csr_row_ptr_d,      this->m.get_row_ptr ( ),      ( num_rows + 1 ) * sizeof (    int ), cudaMemcpyHostToDevice ) );
	SAFE_CALL ( cudaMemcpy ( csr_column_index_d, this->m.get_column_index ( ),              nnz * sizeof (    int ), cudaMemcpyHostToDevice ) );
	SAFE_CALL ( cudaMemcpy ( csr_value_d,        this->m.get_value ( ),                     nnz * sizeof ( double ), cudaMemcpyHostToDevice ) );

	SAFE_CALL ( cusparseCreate ( &handle ) );
	SAFE_CALL ( cusparseCreateMatDescr ( &descr ) );
	SAFE_CALL ( cusparseSetMatType ( descr, this->m.symmetric ( ) ? CUSPARSE_MATRIX_TYPE_SYMMETRIC : CUSPARSE_MATRIX_TYPE_GENERAL ) );
	SAFE_CALL ( cusparseSetMatIndexBase ( descr, CUSPARSE_INDEX_BASE_ZERO ) ); 
}

inline
dcsrmv_operator_gpu::~dcsrmv_operator_gpu ( ) {
	SAFE_CALL ( cudaSetDevice ( dev ) );

	SAFE_CALL ( cusparseDestroyMatDescr ( descr  ) );
	SAFE_CALL ( cusparseDestroy         ( handle ) );

	SAFE_CALL ( cudaFree ( y_d                ) );
	SAFE_CALL ( cudaFree ( x_d                ) );
	SAFE_CALL ( cudaFree ( csr_value_d        ) );
	SAFE_CALL ( cudaFree ( csr_column_index_d ) );
	SAFE_CALL ( cudaFree ( csr_row_ptr_d      ) );
}

inline
void dcsrmv_operator_gpu::operator () ( double const * x, double * y ) const {
	int num_rows    = this->m.get_num_rows    ( );
	int num_columns = this->m.get_num_columns ( );
	int nnz         = this->m.get_nnz         ( );
	double * xx, * yy;
	cudaPointerAttributes attr;

	SAFE_CALL ( cudaSetDevice ( dev ) );
	if ( cudaSuccess == cudaPointerGetAttributes ( &attr, x ) && cudaMemoryTypeDevice == attr.memoryType and dev == attr.device ) {
		xx = ( double * ) x;
	} else {
		SAFE_CALL ( cudaMemcpy ( x_d, x, num_columns * sizeof ( double ), cudaMemcpyDefault ) );
		xx = x_d;
	}
	if ( cudaSuccess == cudaPointerGetAttributes ( &attr, y ) && cudaMemoryTypeDevice == attr.memoryType and dev == attr.device ) {
		yy = y;
	} else {
		yy = y_d;
	}

	double const alpha = 1.0;
	double const  beta = 0.0;
	SAFE_CALL ( cusparseDcsrmv (
		handle,
		CUSPARSE_OPERATION_NON_TRANSPOSE,
		num_rows,
		num_columns,
		nnz,
		&alpha,
		descr,
		csr_value_d,
		csr_row_ptr_d,
		csr_column_index_d,
		xx,
		&beta,
		yy
	) );

	if ( yy != y ) {
		SAFE_CALL ( cudaMemcpy ( y, yy, num_rows * sizeof ( double ), cudaMemcpyDefault ) );
	}
}


/*********************************
 ** dcsrmv_operator_gpu_blocked **
 *********************************/

class dcsrmv_operator_gpu_blocked {
public:
	dcsrmv_operator_gpu_blocked ( class dcsr_matrix_t const & m, int dev = 0, size_t block_size = 0 );
	~dcsrmv_operator_gpu_blocked ( );
	void operator () ( double const * x, double * y ) const;
protected:
	class dcsr_matrix_t const & m;
	size_t block_size;
	int dev;

	int * csr_row_ptr_d;
	double * x_d;
	double * y_d;
	mutable struct buf_t {
		int * csr_column_index_d;
		double * csr_value_d;
		int ib;
		int num_rows;
		cusparseHandle_t handle;
		cusparseMatDescr_t descr;
		cudaStream_t stream;
	} buf[2];
};

inline
dcsrmv_operator_gpu_blocked::dcsrmv_operator_gpu_blocked ( class dcsr_matrix_t const & m, int dev, size_t block_size ) : m ( m ), dev ( dev ), block_size ( block_size ) {
	int num_rows    = this->m.get_num_rows    ( );
	int num_columns = this->m.get_num_columns ( );
	int nnz         = this->m.get_nnz         ( );

	if ( m.symmetric ( ) ) {
		error_exit ( "dcsrmv_operator_mgpu_blocked can only be constructed with a general matrix.\n" );
	}

	SAFE_CALL ( cudaSetDevice ( dev ) );

	if ( !block_size ) {
#define RESERVE_GLOBAL_MEMORY	200 * 1048576llu
		cudaDeviceProp prop;
		cudaGetDeviceProperties ( &prop, dev );
		size_t total_global = prop.totalGlobalMem;
		this->block_size = block_size = ( total_global
			- RESERVE_GLOBAL_MEMORY
			- ( num_rows + 1 ) * sizeof (    int )
			-   num_rows       * sizeof ( double )
			-   num_columns    * sizeof ( double )
		) / 2 / ( sizeof ( int ) + sizeof ( double ) );
	}

	SAFE_CALL ( cudaMalloc ( ( void ** ) &csr_row_ptr_d, ( num_rows + 1 ) * sizeof (    int ) ) );
	SAFE_CALL ( cudaMalloc ( ( void ** ) &x_d,                num_columns * sizeof ( double ) ) );
	SAFE_CALL ( cudaMalloc ( ( void ** ) &y_d,                   num_rows * sizeof ( double ) ) );
	SAFE_CALL ( cudaMemcpy ( csr_row_ptr_d, this->m.get_row_ptr ( ), ( num_rows + 1 ) * sizeof ( int ), cudaMemcpyHostToDevice ) );

	for ( int i = 0; i < 2; ++i ) {
		SAFE_CALL ( cudaMalloc ( ( void ** ) &buf[i].csr_column_index_d, block_size * sizeof (    int ) ) );
		SAFE_CALL ( cudaMalloc ( ( void ** ) &buf[i].csr_value_d,        block_size * sizeof ( double ) ) );
		SAFE_CALL ( cusparseCreate ( &buf[i].handle ) );
		SAFE_CALL ( cusparseCreateMatDescr ( &buf[i].descr ) );
		SAFE_CALL ( cusparseSetMatType ( buf[i].descr, CUSPARSE_MATRIX_TYPE_GENERAL ) );
		SAFE_CALL ( cusparseSetMatIndexBase ( buf[i].descr, CUSPARSE_INDEX_BASE_ZERO ) ); 
		SAFE_CALL ( cudaStreamCreate ( &buf[i].stream ) );
		SAFE_CALL ( cusparseSetStream ( buf[i].handle, buf[i].stream ) );
	}
}

inline
dcsrmv_operator_gpu_blocked::~dcsrmv_operator_gpu_blocked ( ) {
	SAFE_CALL ( cudaSetDevice ( dev ) );
	for ( int i = 0; i < 2; ++i ) {
		SAFE_CALL ( cudaStreamDestroy       ( buf[i].stream ) );
		SAFE_CALL ( cusparseDestroyMatDescr ( buf[i].descr  ) );
		SAFE_CALL ( cusparseDestroy         ( buf[i].handle ) );
		SAFE_CALL ( cudaFree ( buf[i].csr_value_d           ) );
		SAFE_CALL ( cudaFree ( buf[i].csr_column_index_d    ) );
	}
	SAFE_CALL ( cudaFree ( y_d           ) );
	SAFE_CALL ( cudaFree ( x_d           ) );
	SAFE_CALL ( cudaFree ( csr_row_ptr_d ) );
}

inline
void dcsrmv_operator_gpu_blocked::operator () ( double const * x, double * y ) const {
	int num_rows    = this->m.get_num_rows    ( );
	int num_columns = this->m.get_num_columns ( );
	int copy_idx;
	int comp_idx;
	int ib;
	char did_something;

	SAFE_CALL ( cudaSetDevice ( dev ) );
	SAFE_CALL ( cudaMemcpy ( x_d, x, num_columns * sizeof ( double ), cudaMemcpyDefault ) );
	SAFE_CALL ( cudaMemset ( y_d, 0x00, num_rows * sizeof ( double ) ) );

	ib        =  0;
	copy_idx  =  0;
	comp_idx  =  1;
	buf[0].ib = -1;
	buf[1].ib = -1;
	do {
		did_something = 0;
		SAFE_CALL ( cudaDeviceSynchronize ( ) );

		if ( -1 != buf[comp_idx].ib ) {
			struct buf_t * p = &buf[comp_idx];
			int const * row_ptr = m.get_row_ptr ( );
			int offset = row_ptr[p->ib];
			int nnz = row_ptr[p->ib + p->num_rows] - offset;

			double const alpha = 1.0;
			double const  beta = 0.0;
			SAFE_CALL ( cusparseDcsrmv (
				p->handle,
				CUSPARSE_OPERATION_NON_TRANSPOSE,
				p->num_rows,
				num_columns,
				nnz,
				&alpha,
				p->descr,
				p->csr_value_d - offset,
				csr_row_ptr_d + p->ib,
				p->csr_column_index_d - offset,
				x_d,
				&beta,
				y_d + p->ib
			) );

			did_something = 1;
		}

		buf[copy_idx].ib = -1;
		if ( ib != num_rows ) {
			struct buf_t * p = &buf[copy_idx];

			int const * first = m.get_row_ptr ( ) + ib;
			int const * last = m.get_row_ptr ( ) + num_rows;

			int at_least = *first + block_size;
			int const * end = std::lower_bound ( first, last, at_least );

			while ( *end - *first > block_size ) {
				--end;
			}
			if ( *end == *first ) {
				error_exit ( "encountered very long lines.\n" );
			}

			p->ib = ib;
			ib = end - m.get_row_ptr ( );
			p->num_rows = end - first;
			int nnz = *end - *first;
			int offset = *first;
			SAFE_CALL ( cudaMemcpyAsync ( p->csr_value_d,        m.get_value        ( ) + offset, nnz * sizeof ( double ), cudaMemcpyDefault, p->stream ) );
			SAFE_CALL ( cudaMemcpyAsync ( p->csr_column_index_d, m.get_column_index ( ) + offset, nnz * sizeof (    int ), cudaMemcpyDefault, p->stream ) );

			did_something = 1;
		}

		copy_idx = !copy_idx;
		comp_idx = !comp_idx;
	} while ( did_something );

	SAFE_CALL ( cudaDeviceSynchronize ( ) );
	SAFE_CALL ( cudaMemcpy ( y, y_d, num_rows * sizeof ( double ), cudaMemcpyDefault ) );
}


/**************************
 ** dcsrmv_operator_mgpu **
 **************************/

class dcsrmv_operator_mgpu {
public:
	dcsrmv_operator_mgpu ( class dcsr_matrix_t const & m, unsigned mask = ( unsigned ) -1 );
	~dcsrmv_operator_mgpu ( );
	void operator () ( double const * x, double * y ) const;
protected:
	class dcsr_matrix_t const & m;
	struct gpu_args_t {
		int dev;
		int num_rows;
		int nnz;
		cusparseHandle_t handle;
		cusparseMatDescr_t descr;
		double * csr_value_d;
		int * csr_row_ptr_d;
		int * csr_column_index_d;
		int start_row;
		int offset;
		double * x_d;
		double * y_d;
	} * p;
	// devs used, not the total number in the host.
	int ndev;
	// set .num_rows for each gpu_args_t,
	// currently only the average by device stategy is implemented.
	void divide_matrix ( class dcsr_matrix_t const & m, struct gpu_args_t * p, int ndev );
};

inline
void dcsrmv_operator_mgpu::divide_matrix ( class dcsr_matrix_t const & m, struct gpu_args_t * p, int ndev ) {
	int nnz = m.get_nnz ( );
	int const * first = m.get_row_ptr ( );
	int const * last = first + m.get_num_rows ( );
	int block_size = ( nnz + ndev - 1 ) / ndev;
	for ( int i = 0; i < ndev; ++i ) {
		int at_least = block_size * ( i + 1 );
		int const * end = std::lower_bound ( first, last, at_least );
		p[i].num_rows = end - first;
		first = end;
	}
}

inline
dcsrmv_operator_mgpu::dcsrmv_operator_mgpu ( class dcsr_matrix_t const & m, unsigned mask ) : m ( m ) {
	int d, idx;
	int total_devs;
	int num_columns = m.get_num_columns ( );

	if ( m.symmetric ( ) ) {
		error_exit ( "dcsrmv_operator_mgpu can only be constructed with a general matrix.\n" );
	}

	SAFE_CALL ( cudaGetDeviceCount ( &total_devs ) );
	for ( d = ndev = 0; d < total_devs; ++d ) {
		ndev += !!( mask & ( 1 << d ) );
	}
	p = new gpu_args_t[ndev];

	for ( d = idx = 0; d < total_devs; ++d ) {
		if ( !( mask & ( 1 << d ) ) ) continue;
		p[idx++].dev = d;
	}

	divide_matrix ( m, p, ndev );

	int rb = 0, re;
	int const * row_ptr = m.get_row_ptr ( );
	for ( idx = 0; idx < ndev; ++idx, rb = re ) {
		int num_rows = p[idx].num_rows;
		re = rb + num_rows;
		int nnz = row_ptr[re] - row_ptr[rb];
		p[idx].nnz = nnz;
		p[idx].start_row = rb;
		int offset = row_ptr[rb];
		p[idx].offset = offset;

		SAFE_CALL ( cudaSetDevice ( p[idx].dev ) );

		SAFE_CALL ( cudaMalloc ( ( void ** ) &p[idx].csr_row_ptr_d,      ( num_rows + 1 ) * sizeof (    int ) ) );
		SAFE_CALL ( cudaMalloc ( ( void ** ) &p[idx].csr_column_index_d,              nnz * sizeof (    int ) ) );
		SAFE_CALL ( cudaMalloc ( ( void ** ) &p[idx].csr_value_d,                     nnz * sizeof ( double ) ) );
		SAFE_CALL ( cudaMalloc ( ( void ** ) &p[idx].x_d,                     num_columns * sizeof ( double ) ) );
		SAFE_CALL ( cudaMalloc ( ( void ** ) &p[idx].y_d,                        num_rows * sizeof ( double ) ) );
		SAFE_CALL ( cudaMemset (              p[idx].y_d,                  0x00, num_rows * sizeof ( double ) ) );
		SAFE_CALL ( cudaMemcpy ( p[idx].csr_row_ptr_d,      this->m.get_row_ptr ( ) + rb, ( num_rows + 1 ) * sizeof (    int ), cudaMemcpyHostToDevice ) );
		SAFE_CALL ( cudaMemcpy ( p[idx].csr_column_index_d, this->m.get_column_index ( ) + offset,     nnz * sizeof (    int ), cudaMemcpyHostToDevice ) );
		SAFE_CALL ( cudaMemcpy ( p[idx].csr_value_d,        this->m.get_value ( ) + offset,            nnz * sizeof ( double ), cudaMemcpyHostToDevice ) );

		SAFE_CALL ( cusparseCreate ( &p[idx].handle ) );
		SAFE_CALL ( cusparseCreateMatDescr ( &p[idx].descr ) );
		SAFE_CALL ( cusparseSetMatType ( p[idx].descr, CUSPARSE_MATRIX_TYPE_GENERAL ) );
		SAFE_CALL ( cusparseSetMatIndexBase ( p[idx].descr, CUSPARSE_INDEX_BASE_ZERO ) ); 
	}
}

inline
dcsrmv_operator_mgpu::~dcsrmv_operator_mgpu ( ) {
	for ( int idx = 0; idx < ndev; ++idx ) {
		SAFE_CALL ( cudaSetDevice ( p[idx].dev ) );

		SAFE_CALL ( cusparseDestroyMatDescr ( p[idx].descr  ) );
		SAFE_CALL ( cusparseDestroy         ( p[idx].handle ) );

		SAFE_CALL ( cudaFree ( p[idx].y_d                ) );
		SAFE_CALL ( cudaFree ( p[idx].x_d                ) );
		SAFE_CALL ( cudaFree ( p[idx].csr_value_d        ) );
		SAFE_CALL ( cudaFree ( p[idx].csr_column_index_d ) );
		SAFE_CALL ( cudaFree ( p[idx].csr_row_ptr_d      ) );
	}
	delete [] p;
}

inline
void dcsrmv_operator_mgpu::operator () ( double const * x, double * y ) const {
	int num_columns = this->m.get_num_columns ( );

	for ( int idx = 0; idx < ndev; ++idx ) {
		SAFE_CALL ( cudaSetDevice ( p[idx].dev ) );

		SAFE_CALL ( cudaMemcpy ( p[idx].x_d, x, num_columns * sizeof ( double ), cudaMemcpyDefault ) );

		double const alpha = 1.0;
		double const  beta = 0.0;
		SAFE_CALL ( cusparseDcsrmv (
			p[idx].handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			p[idx].num_rows,
			num_columns,
			p[idx].nnz,
			&alpha,
			p[idx].descr,
			p[idx].csr_value_d - p[idx].offset,
			p[idx].csr_row_ptr_d,
			p[idx].csr_column_index_d - p[idx].offset,
			p[idx].x_d,
			&beta,
			p[idx].y_d
		) );
	}
	for ( int idx = 0; idx < ndev; ++idx ) {
		SAFE_CALL ( cudaSetDevice ( p[idx].dev ) );
		SAFE_CALL ( cudaMemcpy ( y + p[idx].start_row, p[idx].y_d, p[idx].num_rows * sizeof ( double ), cudaMemcpyDefault ) );
	}
}


/*****************
 ** mat_v_mkl_t **
 *****************/

struct mat_v_mkl_t {
	mat_v_mkl_t ( double * v, int ldv, int n, int ncv ) : v ( v ), ldv ( ldv ), n ( n ), ncv ( ncv ) { }
	~mat_v_mkl_t ( ) { }

	void set_column ( double * p, int c ) {
		memcpy ( v + c * ldv, p, n * sizeof ( double ) );
	}
	void get_column ( double * p, int c ) {
		memcpy ( p, v + c * ldv, n * sizeof ( double ) );
	}
	void op ( int column, double * ipj, double * irj, double * resid ) {
		cblas_dgemv ( CblasColMajor, CblasTrans, n, column, 1.0, v, ldv, ipj, 1, 0.0, irj, 1 );
		cblas_dgemv ( CblasColMajor, CblasNoTrans, n, column, -1.0, v, ldv, irj, 1, 1.0, resid, 1 );
	}
	void negative_column ( int c ) {
		double * p = v + c * ldv;
		for ( int i = 0; i < n; ++i ) {
			p[i] = -p[i];
		}
	}
	void gemv_notrans ( int column, double alpha, double * x, double beta, double * y ) {
		cblas_dgemv ( CblasColMajor, CblasNoTrans, n, column, alpha, v, ldv, x, 1, beta, y, 1 );
	}
	void gemv_to_column ( int column, double alpha, double * x, double beta, double * y, int c ) {
		cblas_dgemv ( CblasColMajor, CblasNoTrans, n, column, alpha, v, ldv, x, 1, beta, y, 1 );
		memcpy ( v + c * ldv, y, n * sizeof ( double ) );
	}
	void move_columns ( int s, int nc ) {
		memmove ( v, v + s * ldv, nc * ldv * sizeof ( double ) );
	}
	void sync_host ( void ) {
	}
	int get_ld ( void ) {
		return ldv;
	}
	void dorm2r_RN ( int nconv, double * iq, int ldq, double * iw, int * info ) {
		double * workd = new double[n];
		dorm2r ( "Right", "Notranspose", &n, &ncv, &nconv, iq, &ldq, iw, v, &ldv, workd, info );
		delete [] workd;
	}

	double * v;
	int ldv;
	int n;
	int ncv;
};


/*****************
 ** mat_v_gpu_t **
 *****************/

#define PADDING_GRANULARITY	32

#define BS_MOVE_V 256
__global__ void kernel_move_columns ( double * vector_d, int ldv, int n, int np, int nev ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if ( tid >= n ) return;
	int s = tid, d = s + np * ldv;
	for ( int i = 0; i < nev; ++i, s += ldv, d += ldv ) {
		vector_d[s] = vector_d[d];
	}
}
#define BS_NEGATIVE	256
__global__ void kernel_negative ( double * p ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	p[tid] = -p[tid];
}

struct mat_v_gpu_t {
	mat_v_gpu_t ( double * v, int ldv, int n, int ncv, int dev = 0 ) : v ( v ), ldv ( ldv ), n ( n ), ncv ( ncv ), dev ( dev ) {
		SAFE_CALL ( cudaSetDevice ( dev ) );
		ldv_d = CEIL_DIV ( n, PADDING_GRANULARITY ) * PADDING_GRANULARITY;
		SAFE_CALL ( cudaMalloc ( ( void ** ) &v_d,   ncv * ldv_d * sizeof ( double ) ) );
		SAFE_CALL ( cudaMalloc ( ( void ** ) &p_n,             n * sizeof ( double ) ) );
		SAFE_CALL ( cudaMalloc ( ( void ** ) &p_ncv,         ncv * sizeof ( double ) ) );
		SAFE_CALL ( cudaMemset ( p_n,   0x00, n   * sizeof ( double ) ) );
		SAFE_CALL ( cudaMemset ( p_ncv, 0x00, ncv * sizeof ( double ) ) );
		SAFE_CALL ( cublasCreate ( &handle ) );
	}
	~mat_v_gpu_t ( ) {
		SAFE_CALL ( cudaSetDevice ( dev ) );
		SAFE_CALL ( cublasDestroy ( handle ) );
		SAFE_CALL ( cudaFree ( p_ncv ) );
		SAFE_CALL ( cudaFree ( p_n   ) );
		SAFE_CALL ( cudaFree ( v_d   ) );
	}

	void set_column ( double * p, int c ) {
		SAFE_CALL ( cudaSetDevice ( dev ) );
		SAFE_CALL ( cudaMemcpy ( v_d + c * ldv_d, p, n * sizeof ( double ), cudaMemcpyHostToDevice ) );
	}
	void get_column ( double * p, int c ) {
		SAFE_CALL ( cudaSetDevice ( dev ) );
		SAFE_CALL ( cudaMemcpy ( p, v_d + c * ldv_d, n * sizeof ( double ), cudaMemcpyDeviceToHost ) );
	}
	void op ( int column, double * ipj, double * irj, double * resid ) {
		SAFE_CALL ( cudaSetDevice ( dev ) );
		SAFE_CALL ( cudaMemcpy ( p_n, ipj, n * sizeof ( double ), cudaMemcpyHostToDevice ) );
		SAFE_CALL ( cublasDgemv (
			handle, CUBLAS_OP_T,
			n, column,
			&c_d1,
			v_d, ldv_d,
			p_n, 1,
			&c_d0,
			p_ncv, 1
		) );
		SAFE_CALL ( cudaMemcpy ( irj, p_ncv, column * sizeof ( double ), cudaMemcpyDeviceToHost ) );
		SAFE_CALL ( cudaMemcpy ( p_n, resid, n * sizeof ( double ), cudaMemcpyHostToDevice ) );
		SAFE_CALL ( cublasDgemv (
			handle, CUBLAS_OP_N,
			n, column,
			&c_dn1,
			v_d, ldv_d,
			p_ncv, 1,
			&c_d1,
			p_n, 1
		) );
		SAFE_CALL ( cudaMemcpy ( resid, p_n, n * sizeof ( double ), cudaMemcpyDeviceToHost ) );
	}
	void negative_column ( int c ) {
		SAFE_CALL ( cudaSetDevice ( dev ) );
		kernel_negative <<< CEIL_DIV ( n, BS_NEGATIVE ), BS_NEGATIVE >>> ( v_d + c * ldv );
	}
	void gemv_notrans ( int column, double alpha, double * x, double beta, double * y ) {
		SAFE_CALL ( cudaSetDevice ( dev ) );
		SAFE_CALL ( cudaMemcpy ( p_ncv, x, column * sizeof ( double ), cudaMemcpyHostToDevice ) );
		SAFE_CALL ( cudaMemcpy ( p_n, y, n * sizeof ( double ), cudaMemcpyHostToDevice ) );
		SAFE_CALL ( cublasDgemv (
			handle, CUBLAS_OP_N,
			n, column,
			&alpha,
			v_d, ldv_d,
			p_ncv, 1,
			&beta,
			p_n, 1
		) );
		SAFE_CALL ( cudaMemcpy ( y, p_n, n * sizeof ( double ), cudaMemcpyDeviceToHost ) );
	}
	void gemv_to_column ( int column, double alpha, double * x, double beta, double * y, int c ) {
		SAFE_CALL ( cudaSetDevice ( dev ) );
		SAFE_CALL ( cudaMemcpy ( p_ncv, x, column * sizeof ( double ), cudaMemcpyHostToDevice ) );
		SAFE_CALL ( cudaMemcpy ( p_n, y, n * sizeof ( double ), cudaMemcpyHostToDevice ) );
		SAFE_CALL ( cublasDgemv (
			handle, CUBLAS_OP_N,
			n, column,
			&alpha,
			v_d, ldv_d,
			p_ncv, 1,
			&beta,
			p_n, 1
		) );
		SAFE_CALL ( cudaMemcpy ( v_d + c * ldv_d, p_n, n * sizeof ( double ), cudaMemcpyDeviceToDevice ) );
	}
	void move_columns ( int s, int nc ) {
		SAFE_CALL ( cudaSetDevice ( dev ) );
		kernel_move_columns <<< CEIL_DIV ( n, BS_MOVE_V ), BS_MOVE_V >>> ( v_d, ldv_d, n, s, nc );
	}
	void sync_host ( void ) {
		SAFE_CALL ( cudaSetDevice ( dev ) );
		SAFE_CALL ( cudaMemcpy2D (
			v, ldv * sizeof ( double ),
			v_d, ldv_d * sizeof ( double ),
			n * sizeof ( double ), ncv,
			cudaMemcpyDeviceToHost
		) );
	}
	int get_ld ( void ) {
		return ldv;
	}
	void dorm2r_RN ( int nconv, double * iq, int ldq, double * iw, int * info ) {
		double const alpha = 1.0;
		double const  beta = 0.0;
		for ( int i = 0; i < nconv; ++i ) {
			int ni = ncv - i;
			int jc = i;
			double * p = iq + i + i * ldq;
			double qii = *p;
			*p = 1.0;
			if ( 0.0 != iw[i] ) {
				SAFE_CALL ( cudaMemcpy ( p_ncv, p, ni * sizeof ( double ), cudaMemcpyHostToDevice ) );
				SAFE_CALL ( cublasDgemv (
					handle, CUBLAS_OP_N,
					n, ni,
					&alpha,
					v_d + jc * ldv_d, ldv_d,
					p_ncv, 1,
					&beta,
					p_n, 1
				) );
				double d_1 = -iw[i];
				SAFE_CALL ( cublasDger (
					handle,
					n, ni,
					&d_1,
					p_n, 1,
					p_ncv, 1,
					v_d + jc * ldv_d, ldv_d
				) );
			}
			*p = qii;
		}
	}

	cublasHandle_t handle;
	double * v;
	double * v_d;
	double * p_n;
	double * p_ncv;
	int ldv;
	int ldv_d;
	int n;
	int ncv;
	int dev;
};


/*************************
 ** mat_v_gpu_blocked_t **
 *************************/

struct mat_v_gpu_blocked_t {
	mat_v_gpu_blocked_t ( double * v, int ldv, int n, int ncv, int dev = 0, int buf_size = 0 ) : v ( v ), ldv ( ldv ), n ( n ), ncv ( ncv ), dev ( dev ), buf_size ( buf_size ) {
		SAFE_CALL ( cudaSetDevice ( dev ) );
		SAFE_CALL ( cudaMalloc ( ( void ** ) &p_n,             n * sizeof ( double ) ) );
		SAFE_CALL ( cudaMalloc ( ( void ** ) &p_ncv,         ncv * sizeof ( double ) ) );
		SAFE_CALL ( cudaMemset ( p_n,   0x00, n   * sizeof ( double ) ) );
		SAFE_CALL ( cudaMemset ( p_ncv, 0x00, ncv * sizeof ( double ) ) );

		if ( !buf_size ) {
#define RESERVE_GLOBAL_MEMORY	200 * 1048576llu
			cudaDeviceProp prop;
			cudaGetDeviceProperties ( &prop, dev );
			size_t total_global = prop.totalGlobalMem;
			this->buf_size = buf_size = ( total_global
				- RESERVE_GLOBAL_MEMORY
				- n   * sizeof ( double )
				- ncv * sizeof ( double )
			) / 2;
		}
		for ( int i = 0; i < 2; ++i ) {
			SAFE_CALL ( cudaMalloc ( ( void ** ) &buf[i].buf, buf_size ) );
			SAFE_CALL ( cublasCreate ( &buf[i].handle ) );
			SAFE_CALL ( cudaStreamCreate ( &buf[i].stream ) );
			SAFE_CALL ( cublasSetStream ( buf[i].handle, buf[i].stream ) );
		}

		SAFE_CALL ( cublasCreate ( &handle ) );
	}
	~mat_v_gpu_blocked_t ( ) {
		SAFE_CALL ( cudaSetDevice ( dev ) );
		for ( int i = 0; i < 2; ++i ) {
			SAFE_CALL ( cudaFree ( buf[i].buf ) );
			SAFE_CALL ( cublasDestroy ( buf[i].handle ) );
			SAFE_CALL ( cudaStreamDestroy ( buf[i].stream ) );
		}
		SAFE_CALL ( cublasDestroy ( handle ) );
		SAFE_CALL ( cudaFree ( p_ncv ) );
		SAFE_CALL ( cudaFree ( p_n   ) );
	}
	void set_column ( double * p, int c ) {
		memcpy ( v + c * ldv, p, n * sizeof ( double ) );
	}
	void get_column ( double * p, int c ) {
		memcpy ( p, v + c * ldv, n * sizeof ( double ) );
	}
	void negative_column ( int c ) {
		double * p = v + c * ldv;
		for ( int i = 0; i < n; ++i ) {
			p[i] = -p[i];
		}
	}
	void op ( int column, double * ipj, double * irj, double * resid ) {
		SAFE_CALL ( cudaSetDevice ( dev ) );
		SAFE_CALL ( cudaMemcpy ( p_n, ipj, n * sizeof ( double ), cudaMemcpyHostToDevice ) );
		gemv_dev_trans ( column, 1.0, 0.0 );
		SAFE_CALL ( cudaMemcpy ( irj, p_ncv, column * sizeof ( double ), cudaMemcpyDeviceToHost ) );
		SAFE_CALL ( cudaMemcpy ( p_n, resid, n * sizeof ( double ), cudaMemcpyHostToDevice ) );
		gemv_dev_notrans ( column, -1.0, 1.0 );
		SAFE_CALL ( cudaMemcpy ( resid, p_n, n * sizeof ( double ), cudaMemcpyDeviceToHost ) );
	}
	void gemv_notrans ( int column, double alpha, double * x, double beta, double * y ) {
		SAFE_CALL ( cudaSetDevice ( dev ) );
		SAFE_CALL ( cudaMemcpy ( p_ncv, x, column * sizeof ( double ), cudaMemcpyHostToDevice ) );
		SAFE_CALL ( cudaMemcpy ( p_n, y, n * sizeof ( double ), cudaMemcpyHostToDevice ) );
		gemv_dev_notrans ( column, alpha, beta );
		SAFE_CALL ( cudaMemcpy ( y, p_n, n * sizeof ( double ), cudaMemcpyDeviceToHost ) );
	}
	void gemv_to_column ( int column, double alpha, double * x, double beta, double * y, int c ) {
		SAFE_CALL ( cudaSetDevice ( dev ) );
		SAFE_CALL ( cudaMemcpy ( p_ncv, x, column * sizeof ( double ), cudaMemcpyHostToDevice ) );
		SAFE_CALL ( cudaMemcpy ( p_n, y, n * sizeof ( double ), cudaMemcpyHostToDevice ) );
		gemv_dev_notrans ( column, alpha, beta );
		SAFE_CALL ( cudaMemcpy ( v + c * ldv, p_n, n * sizeof ( double ), cudaMemcpyDeviceToHost ) );
	}
	void move_columns ( int s, int nc ) {
		memmove ( v, v + s * ldv, nc * ldv * sizeof ( double ) );
	}
	double * get_base ( void ) {
		return v;
	}
	int get_ld ( void ) {
		return ldv;
	}
protected:
	void gemv_dev_trans ( int column, double alpha, double beta ) {
		double * x_d = p_n;
		double * y_d = p_ncv;
		int ldv_d = CEIL_DIV ( n, 32 ) * 32;
		int max_bs = buf_size / ( ldv_d * sizeof ( double ) );
		if ( !max_bs ) {
			error_exit ( "encountered very large matrix.\n" );
		}
		int ib = 0;
		char did_something;
		buf[0].ib = -1;
		buf[1].ib = -1;
		int copy_idx = 0;
		int comp_idx = 1;

		SAFE_CALL ( cudaSetDevice ( dev ) );
		do {
			did_something = 0;
			SAFE_CALL ( cudaDeviceSynchronize ( ) );

			if ( -1 != buf[comp_idx].ib ) {
				struct buf_t * p = &buf[comp_idx];
				SAFE_CALL ( cublasDgemv (
					p->handle, CUBLAS_OP_T,
					n, p->bs,
					&alpha,
					p->buf, ldv_d,
					x_d, 1,
					&beta,
					y_d + p->ib, 1
				) );
				did_something = 1;
			}

			buf[copy_idx].ib = -1;
			if ( ib != column ) {
				int bs = column - ib;
				if ( bs > max_bs ) bs = max_bs;
				struct buf_t * p = &buf[copy_idx];
				p->ib = ib;
				p->bs = bs;
				SAFE_CALL ( cudaMemcpy2DAsync (
					p->buf, ldv_d * sizeof ( double ),
					v + ib * ldv, ldv * sizeof ( double ),
					n * sizeof ( double ), bs,
					cudaMemcpyHostToDevice, p->stream
				) );
				ib += bs;
				did_something = 1;
			}

			copy_idx = !copy_idx;
			comp_idx = !comp_idx;
		} while ( did_something );
	}
	void gemv_dev_notrans ( int column, double alpha, double beta ) {
		double * x_d = p_ncv;
		double * y_d = p_n;
		int max_bs = buf_size / ( 32 * column * sizeof ( double ) ) * 32;
		if ( !max_bs ) {
			error_exit ( "encountered very large matrix.\n" );
		}
		int ib = 0;
		char did_something;
		buf[0].ib = -1;
		buf[1].ib = -1;
		int copy_idx = 0;
		int comp_idx = 1;

		SAFE_CALL ( cudaSetDevice ( dev ) );
		do {
			did_something = 0;
			SAFE_CALL ( cudaDeviceSynchronize ( ) );

			if ( -1 != buf[comp_idx].ib ) {
				struct buf_t * p = &buf[comp_idx];
				SAFE_CALL ( cublasDgemv (
					p->handle, CUBLAS_OP_N,
					p->bs, column,
					&alpha,
					p->buf, max_bs,
					x_d, 1,
					&beta,
					y_d + p->ib, 1
				) );
				did_something = 1;
			}

			buf[copy_idx].ib = -1;
			if ( ib != n ) {
				int bs = n - ib;
				if ( bs > max_bs ) bs = max_bs;
				struct buf_t * p = &buf[copy_idx];
				p->ib = ib;
				p->bs = bs;
				SAFE_CALL ( cudaMemcpy2DAsync (
					p->buf, max_bs * sizeof ( double ),
					v + ib, ldv * sizeof ( double ),
					bs * sizeof ( double ), column,
					cudaMemcpyHostToDevice, p->stream
				) );
				ib += bs;
				did_something = 1;
			}

			copy_idx = !copy_idx;
			comp_idx = !comp_idx;
		} while ( did_something );
	}

	struct buf_t {
		double * buf;
		int ib;
		int bs;
		cublasHandle_t handle;
		cudaStream_t stream;
	} buf[2];
	int buf_size;

	cublasHandle_t handle;
	double * v;
	double * p_ncv;
	double * p_n;
	int ldv;
	int n;
	int ncv;
	int dev;
};


/***************
 ** ds_solver **
 ***************/

struct d2_t {
	double d1;
	double d2;
	bool operator < ( d2_t const & o ) const { return d1 < o.d1; }
	bool operator > ( d2_t const & o ) const { return d1 > o.d1; }
};

inline bool greater_by_d2 ( struct d2_t const & l, struct d2_t const & r ) {
	return l.d2 > r.d2;
}

inline bool less_by_d2 ( struct d2_t const & l, struct d2_t const & r ) {
	return l.d2 < r.d2;
}

static inline
void dgetv0 ( int n, double * resid, double & rnorm ) {
	int iseed[4] = { 1, 3, 5, 7 };
	int idist = 2;
	dlarnv ( &idist, iseed, &n, resid );
	rnorm = cblas_dnrm2 ( n, resid, 1 );
}

static inline
void dsconv ( int n, double * ritz, double * bounds, double tol, int & nconv ) {
	double eps23 = pow ( dlamch ( "E" ), c_23 );
	nconv = 0;
	for ( int i = 0; i < n; ++i ) {
		double temp = MAX ( eps23, abs ( ritz[i] ) );
		nconv += bounds[i] <= tol * temp;
	}
}

static inline
void dsgets ( int nev, int np, double * ritz, double * bounds, double * shifts ) {
	int ncv = nev + np;
	struct d2_t * p = new d2_t[ncv];
	for ( int i = 0; i < ncv; ++i ) {
		p[i].d1 = ritz[i];
		p[i].d2 = bounds[i];
	}
	std::sort ( p, p + ncv );
	std::sort ( p, p + np, greater_by_d2 );
	for ( int i = 0; i < ncv; ++i ) {
		ritz[i] = p[i].d1;
		bounds[i] = p[i].d2;
	}
	delete [] p;
}

template < typename dcsrmv_operator_t, typename mat_v_t >
void dsaitr ( int n, int k, int np, double * resid, double & rnorm, mat_v_t & mat_v, double * h, int ldh, dcsrmv_operator_t const & av ) {
	double safmin = dlamch ( "S" );
	double * ipj = new double[n];
	double * irj = new double[n];

	// ARNOLDI ITERATION LOOP
	for ( int j = k; j < k + np; ++j ) {
		// STEP 2
		if ( rnorm > safmin ) {
			double temp1 = 1.0 / rnorm;
			for ( int i = 0; i < n; ++i ) {
				irj[i] = resid[i] * temp1;
			}
		} else {
			memcpy ( irj, resid, n * sizeof ( double ) );
		}

		mat_v.set_column ( irj, j );

		// STEP 3
		av ( irj, resid );
		memcpy ( irj, resid, sizeof ( double ) * n );

		// STEP 4
		memcpy ( ipj, resid, sizeof ( double ) * n );
		double wnorm = cblas_dnrm2 ( n, resid, 1 );

		mat_v.op ( j + 1, ipj, irj, resid );

		h[ldh + j] = irj[j];
		if ( 0 == j ) {
			h[j] = 0;
		} else {
			h[j] = rnorm;
		}
		memcpy ( ipj, resid, sizeof ( double ) * n );
		rnorm = cblas_dnrm2 ( n, resid, 1 );

		// STEP 5
		int iter = 0;
		while ( rnorm < wnorm * 0.717 ) {
			mat_v.op ( j + 1, ipj, irj, resid );

			h[ldh + j] += irj[j];
			memcpy ( ipj, resid, sizeof ( double ) * n );

			double rnorm1 = cblas_dnrm2 ( n, resid, 1 );
			if ( rnorm1 > rnorm * 0.717 ) {
				rnorm = rnorm1;
			} else {
				rnorm = rnorm1;
				++iter;
				if ( iter <= 1 ) {
					continue;
				}
				for ( int jj = 0; jj < n; ++jj ) {
					resid[jj] = 0.0;
				}
				rnorm = 0.0;
			}
			break;
		}

		if ( h[j] < 0.0 ) {
			h[j] = -h[j];
			if ( j < k + np - 1 ) {
				mat_v.negative_column ( j + 1 );
			} else {
				for ( int i = 0; i < n; ++i ) {
					resid[i] *= -1.0;
				}
			}
		}

	}

	delete [] irj;
	delete [] ipj;
}

template < typename mat_v_t >
void dsapps( int n, int nev, int np, double * shift, mat_v_t & mat_v, double * h, int ldh, double * resid ) {
	double epsmch = dlamch ( "E" );
	int ncv = nev + np;
	int ldq = ncv;
	double * q = new double[ncv * ncv];
	double * workd = new double[n * 2];

	dlaset ( "All", &ncv, &ncv, &c_d0, &c_d1, q, &ncv );
	if ( 0 == np ) return;

	int itop = 1;
	double c = 0.0, s = 0.0, r = 0.0;
	for ( int jj = 1; jj <= np; ++jj ) {
		int istart = itop;

		while ( 1 ) {
			int iend = ncv;
			for ( int i = istart; i <= ncv - 1; ++i ) {
				double big = abs ( h[ldh + i - 1] ) + abs ( h[ldh + i] );
				if ( h[i] <= epsmch * big ) {
					h[i] = 0.0;
					iend = i;
					break;
				}
			}

			if ( istart < iend ) {
				double f = h[istart + ldh - 1] - shift[jj - 1];
				double g = h[istart];
				dlartg ( &f, &g, &c, &s, &r );

				double a1 = c * h[istart + ldh - 1] + s * h[istart];
				double a2 = c * h[istart] + s * h[istart + ldh];
				double a4 = c * h[istart + ldh] - s * h[istart];
				double a3 = c * h[istart] - s * h[istart + ldh - 1];
				h[istart + ldh - 1] = c * a1 + s * a2;
				h[istart + ldh] = c * a4 - s * a3;
				h[istart] = c * a3 + s * a4;

				int i2 = MIN ( istart + jj, ncv );
				for ( int j = 1; j <= i2; ++j ) {
					a1 = c * q[j + ( istart - 1 ) * ldq - 1] + s * q[j + istart * ldq - 1];
					q[j + istart * ldq - 1] = -s * q[j + ( istart - 1 ) * ldq - 1] + c * q[j + istart * ldq - 1];
					q[j + ( istart - 1 ) * ldq - 1] = a1;
				}

				for ( int i = istart + 1; i <= iend - 1; ++i ) {
					f = h[i - 1];
					g = s * h[i];

					h[i] = c * h[i];
					dlartg ( &f, &g, &c, &s, &r );

					if ( r < 0.0 ) {
						r = -r;
						c = -c;
						s = -s;
					}
					h[i - 1] = r;

					double a1 = c * h[i + ldh - 1] + s * h[i];
					double a2 = c * h[i] + s * h[i + ldh];
					double a3 = c * h[i] - s * h[i + ldh - 1];
					double a4 = c * h[i + ldh] - s * h[i];
					h[i + ldh - 1] = c * a1 + s * a2;
					h[i + ldh] = c * a4 - s * a3;
					h[i] = c * a3 + s * a4;

					int i3 = MIN ( i + jj, ncv );
					for ( int j = 1; j <= i3; ++j ) {
						double a1 = c * q[j + ( i - 1 ) * ldq - 1] + s * q[j + i * ldq - 1];
						q[j + i * ldq - 1] = -s * q[j + ( i - 1 ) * ldq - 1] + c * q[j + i * ldq - 1];
						q[j + ( i - 1 ) * ldq - 1] = a1;
					}
				}
			}
			istart = iend + 1;

			if ( h[iend - 1] < 0.0 ) {
				h[iend - 1] = -h[iend - 1];
				for ( int i = 0; i < ncv; ++i ) {
					q[( iend - 1 ) * ldq + i] *= -1.0;
				}
			}

			if ( iend < ncv ) {
				continue;
			}

			for ( int i = itop; i <= ncv - 1; ++i ) {
				if ( h[i] > 0.0 ) {
					break;
				}
				++itop;
			}
			break;
		}
	}

	for ( int i = itop; i <= ncv - 1; ++i ) {
		double big = abs ( h[i + ldh - 1] ) + abs ( h[i + ldh] );
		if ( h[i] <= epsmch * big ) {
			h[i] = 0.0;
		}
	}

	if ( h[nev] > 0 ) {
		mat_v.gemv_notrans ( ncv, 1.0, q + nev * ldq, 0.0, workd + n );
	}

	for ( int i = 1; i <= nev; ++i ) {
		mat_v.gemv_to_column ( ncv - i + 1, 1.0, q + ( nev - i ) * ldq, 0.0, workd, ncv - i );
	}

	mat_v.move_columns ( np, nev );

	if ( h[nev] > 0.0 ) {
		mat_v.set_column ( workd + n, nev );
	}

	for ( int i = 0; i < n; ++i ) {
		resid[i] *= q[ncv + ( nev - 1 ) * ldq - 1];
	}

	if ( h[nev] > 0.0 ) {
		mat_v.get_column ( workd, nev );
		for ( int i = 0; i < n; ++i ) {
			resid[i] += h[nev] * workd[i];
		}
	}

	delete [] workd;
	delete [] q;
}

struct cmp_t {
	cmp_t ( double * p ) : p ( p ) { }
	double * p;
	bool operator () ( int l, int r ) const { return p[l] < p[r]; }
};
static inline
void dsesrt ( int nconv, double * value, int ncv, double * iq, int ldq ) {
	int * idx = new int[nconv];
	double * qq = new double[ldq * nconv];

	for ( int i = 0; i < nconv; ++i ) {
		idx[i] = i;
	}
	std::sort ( idx + 0, idx + nconv, cmp_t ( value ) );

	memcpy ( qq, value, sizeof ( double ) * nconv );
	for ( int i = 0; i < nconv; ++i ) {
		value[i] = qq[idx[i]];
	}

	memcpy ( qq, iq, sizeof ( double ) * ldq * nconv );
	for ( int c = 0; c < nconv; ++c ) {
		memcpy ( iq + c * ldq, qq + idx[c] * ldq, sizeof ( double ) * ncv );
	}

	delete [] qq;
	delete [] idx;
}

#include "timing.h"

template < typename dcsrmv_operator_t, typename mat_v_t >
int ds_solver ( int n, int nev, int ncv, double * value, dcsrmv_operator_t const & av, mat_v_t & mat_v ) {
TIMING_BEGIN;
	double tol = dlamch ( "E" );
	double eps23 = pow ( dlamch ( "E" ), c_23 );

	int np = ncv - nev;
	int nconv = 0;

	int nev0 = nev;
	int np0 = np;

	double rnorm;
	double * resid = new double[n];
	double * workd = new double[n];

	dgetv0 ( n, resid, rnorm );

	int lworkl = ncv * (ncv + 8);
	double * workl_alloc = new double[lworkl];
	memset ( workl_alloc, 0x00, sizeof ( double ) * lworkl );

	double * h = workl_alloc;
	int ldh = ncv;
	double * ritz = h + ncv * 2;
	double * bounds = ritz + ncv;
	double * q = bounds + ncv;
	int ldq = ncv;
	double * workl = q + ncv * ncv;

	int info;

	dsaitr ( n, 0, nev0, resid, rnorm, mat_v, h, ldh, av );

#define MAX_ITER	200
	int iter;
	for ( iter = 0; iter < MAX_ITER; ++iter ) {
		dsaitr ( n, nev, np, resid, rnorm, mat_v, h, ldh, av );

		memcpy ( ritz, h + ldh, sizeof ( double ) * ncv );
		memcpy ( workl, h + 1, sizeof ( double ) * ( ncv - 1 ) );

		info = 0;
		dsteqr ( "I", &ncv, ritz, workl, q, &ncv, workl + ncv, &info );
		if ( info ) return 1;
		for ( int i = 0; i < ncv; ++i ) {
			bounds[i] = rnorm * abs ( q[i * ncv + ncv - 1] );
		}

		memcpy ( workl + ncv,     ritz,   sizeof ( double ) * ncv );
		memcpy ( workl + ncv * 2, bounds, sizeof ( double ) * ncv );

		nev = nev0;
		np = np0;

		dsgets ( nev, np, ritz, bounds, NULL );

		memcpy ( workl + np, bounds + np, sizeof ( double ) * nev );
		dsconv ( nev, ritz + np, workl + np, tol, nconv );

		for ( int j = 0; j < np; ++j ) {
			if ( 0.0 == bounds[j] ) {
				--np;
				++nev;
			}
		}

		if ( nconv >= nev0 || 0 == np ) {
			break;
		}

		int nevbef = nev;
		nev += MIN ( nconv, np / 2 );
		if ( 1 == nev && ncv >= 6 ) {
			nev = ncv / 2;
		} else if ( 1 == nev && ncv > 2 ) {
			nev = 2;
		}
		np = ncv - nev;

		if ( nevbef < nev ) {
			dsgets ( nev, np, ritz, bounds, workl );
		}

		dsapps(n, nev, np, ritz, mat_v, h, ldh, resid );

		memcpy ( workd, resid, sizeof ( double ) * n );
		rnorm = cblas_dnrm2 ( n, resid, 1 );
	}

	struct d2_t * p = new d2_t[ncv];
	for ( int i = 0; i < ncv; ++i ) {
		p[i].d1 = ritz[i];
		p[i].d2 = bounds[i];
	}
	std::sort ( p, p + ncv, std::greater < d2_t > ( ) );
	for ( int j = 0; j < nev0; ++j ) {
		double temp = MAX ( eps23, abs ( p[j].d1 ) );
		p[j].d2 /= temp;
	}
	std::sort ( p, p + nev0, less_by_d2 );
	for ( int j = 0; j < nev0; ++j ) {
		double temp = MAX ( eps23, abs ( p[j].d1 ) );
		p[j].d2 *= temp;
	}
	std::sort ( p, p + nconv, std::less < d2_t > ( ) );
	for ( int i = 0; i < ncv; ++i ) {
		ritz[i] = p[i].d1;
		bounds[i] = p[i].d2;
	}
	delete [] p;

	h[0] = rnorm;

	if ( nconv < nev0 ) return 1;
TIMING_END ( 0 );
///////////////////// for eig vectors ////////////////////

	nev = nev0;
	np = np0;

	char * select = new char[ncv];
	for ( int j = 0; j < ncv; ++j ) {
		bounds[j] = j + 1;
		select[j] = 0;
	}

	int irz = 5 * ncv + ncv * ncv;
	int ibd = irz + ncv;

	dsgets( nev, np, workl_alloc + irz, bounds, h );

	int reord = 0;
	int numcnv = 0;
	for ( int j = 0; j < ncv; ++j ) {
		double temp1 = MAX ( eps23, abs ( workl_alloc[irz + ncv - j - 1] ) );
		int jj = bounds[ncv - j - 1];
		if ( numcnv < nconv && workl_alloc[ibd + jj - 1] <= tol * temp1 ) {
			select[jj - 1] = 1;
			++numcnv;
			if ( jj > nev ) reord = 1;
		}
	}

	if ( numcnv != nconv ) {
		return 1;
	}

	double * ihd = workl_alloc + ncv * 4;
	double * ihb = workl_alloc + ncv * 5;

	memcpy ( ihb, h +   1, sizeof ( double ) * ( ncv - 1 ) );
	memcpy ( ihd, h + ldh, sizeof ( double ) *   ncv       );

	double * iq = workl_alloc + ncv * 6;
	double * iw = iq + ncv * ncv;

	info = 0;
	dsteqr ( "I", &ncv, ihd, ihb, iq, &ldq, iw, &info );
	if ( info ) return 1;

	if ( reord ) {
		int left = 0, right = ncv - 1;
		while ( 1 ) {
			while ( select[left] ) {
				++left;
			}
			while ( !select[right] ) {
				--right;
			}
			if ( left > right ) {
				break;
			}
			double temp = ihd[left];
			ihd[left] = ihd[right];
			ihd[right] = temp;

			memcpy ( iw, iq + ldq * left, sizeof ( double ) * ncv );
			memcpy ( iq + ldq * left, iq + ldq * right, sizeof ( double ) * ncv );
			memcpy ( iq + ldq * right, iw, sizeof ( double ) * ncv );

			++left;
			--right;
		}
	}

	memcpy ( value, ihd, sizeof ( double ) * nconv );
	dsesrt ( nconv, value, ncv, iq, ldq );

	info = 0;
	dgeqr2 ( &ncv, &nconv, iq, &ldq, iw, ihb, &info );
	if ( info ) return 1;

	mat_v.dorm2r_RN ( nconv, iq, ldq, iw, &info );
	if ( info ) return 1;

TIMING_END ( 1 );
TIMING_PRINT ( "au+av", 0 );
TIMING_PRINT ( "eu", 1 );

	mat_v.sync_host ( );

	delete [] select;
	delete [] workl_alloc;
	delete [] workd;
	delete [] resid;
	return 0;
}


/*******************
 ** ds_solver_mkl **
 *******************/

template < typename dcsrmv_operator_t >
int ds_solver_mkl ( int n, int nev, int ncv, double * value, double * vector, int ldv, dcsrmv_operator_t const & dcsrmv_operator ) {
	mat_v_mkl_t mat_v ( vector, ldv, n, ncv );
	return ds_solver ( n, nev, ncv, value, dcsrmv_operator, mat_v );
}


/*******************
 ** ds_solver_gpu **
 *******************/

template < typename dcsrmv_operator_t >
int ds_solver_gpu ( int n, int nev, int ncv, double * value, double * vector, int ldv, dcsrmv_operator_t const & dcsrmv_operator, int dev = 0 ) {
	mat_v_gpu_t mat_v ( vector, ldv, n, ncv, dev );
	return ds_solver ( n, nev, ncv, value, dcsrmv_operator, mat_v );
}


/***************************
 ** ds_solver_gpu_blocked **
 ***************************/

template < typename dcsrmv_operator_t >
int ds_solver_gpu_blocked ( int n, int nev, int ncv, double * value, double * vector, int ldv, dcsrmv_operator_t const & dcsrmv_operator, int dev = 0 ) {
	mat_v_gpu_blocked_t mat_v ( vector, ldv, n, ncv, dev );
	return ds_solver ( n, nev, ncv, value, dcsrmv_operator, mat_v );
}


} // namespace HLanc
