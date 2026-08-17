#include "matrix_operation.h"
#include <immintrin.h>

Matrix MatrixOperation::NaiveMatMul(const Matrix &A, const Matrix &B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}
	
	
	Matrix C(n,m);
	
	for(int i = 0; i < n ; i++) {
		for (int j = 0 ; j< m ; j++) {
			for(int l = 0; l < k; l++) {
				C(i,j) += A(i,l) * B(l,j);
			}
		}
	}
	
	return C;
}

// Loop reordered matrix multiplication (ikj order for better cache locality)
Matrix MatrixOperation::ReorderedMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}
	
	
	Matrix C(n,m);

//----------------------------------------------------- Write your code here ----------------------------------------------------------------
	for (size_t i = 0; i < n; i++) {
		for (size_t l = 0; l < k; l++) {
			element_t a_il = A(i, l);
			for (size_t j = 0; j < m; j++) {
				C(i, j) += a_il * B(l, j);
			}
		}
	}
//-------------------------------------------------------------------------------------------------------------------------------------------


	return C;
}

// Loop unrolled matrix multiplication
Matrix MatrixOperation::UnrolledMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);

    const int UNROLL = 4;
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            // Four independent accumulators break the true dependency chain
            // on `sum`, so the additions can issue back-to-back instead of
            // each waiting on the previous one to retire.
            element_t sum0 = INITIAL_VALUE, sum1 = INITIAL_VALUE;
            element_t sum2 = INITIAL_VALUE, sum3 = INITIAL_VALUE;
            size_t l = 0;
            for (; l + UNROLL <= k; l += UNROLL) {
                sum0 += A(i, l)     * B(l, j);
                sum1 += A(i, l + 1) * B(l + 1, j);
                sum2 += A(i, l + 2) * B(l + 2, j);
                sum3 += A(i, l + 3) * B(l + 3, j);
            }
            element_t sum = sum0 + sum1 + sum2 + sum3;
            for (; l < k; l++) {
                sum += A(i, l) * B(l, j);
            }
            C(i, j) = sum;
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// Tiled (blocked) matrix multiplication for cache efficiency
Matrix MatrixOperation::TiledMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);
    const int T = 128;   // tile size
	int i_max = 0;
	int k_max = 0;
	int j_max = 0;
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for (int ii = 0; ii < (int)n; ii += T) {
        i_max = std::min(ii + T, (int)n);
        for (int kk = 0; kk < (int)k; kk += T) {
            k_max = std::min(kk + T, (int)k);
            for (int jj = 0; jj < (int)m; jj += T) {
                j_max = std::min(jj + T, (int)m);
                // Each T*T tile is 128*128*8B = 128KiB; three tiles (A,B,C)
                // in flight at once is ~384KiB, sized to fit L2 -- matching
                // the tile-size reasoning already validated on this
                // hardware in Part 1's mat_mul.c.
                for (int i = ii; i < i_max; i++) {
                    for (int l = kk; l < k_max; l++) {
                        element_t a_il = A(i, l);
                        for (int j = jj; j < j_max; j++) {
                            C(i, j) += a_il * B(l, j);
                        }
                    }
                }
            }
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// SIMD vectorized matrix multiplication (using AVX2)
Matrix MatrixOperation::VectorizedMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    // 256-bit AVX2 lanes hold 4 doubles; no -mfma in the Makefile, so this
    // is a plain multiply + add rather than a fused multiply-add.
    for (size_t i = 0; i < n; i++) {
        for (size_t l = 0; l < k; l++) {
            __m256d a_vec = _mm256_set1_pd(A(i, l));
            size_t j = 0;
            for (; j + 4 <= m; j += 4) {
                __m256d b_vec = _mm256_loadu_pd(&B(l, j));
                __m256d c_vec = _mm256_loadu_pd(&C(i, j));
                c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_vec, b_vec));
                _mm256_storeu_pd(&C(i, j), c_vec);
            }
            for (; j < m; j++) {
                C(i, j) += A(i, l) * B(l, j);
            }
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// Optimized matrix transpose
Matrix MatrixOperation::Transpose(const Matrix& A) {
	size_t rows = A.getRows();
	size_t cols = A.getCols();
	Matrix result(cols, rows);

	// for (size_t i = 0; i < rows; ++i) {
	// 	for (size_t j = 0; j < cols; ++j) {
	// 		result(j, i) = A(i, j);
	// 	}
	// }

	// Optimized transpose using blocking for better cache performance
	// This is a simple implementation, more advanced techniques can be applied
	// Write your code here and commnent the above code
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
	// The naive version reads A row-major (cache-friendly) but writes
	// result column-major (one cache line touched per element, evicted
	// before its neighbour is written) -- for a 512x512 double matrix
	// that's a full stride-4096B scatter. Blocking keeps both the read
	// tile and the write tile within one cache-sized region at a time.
	const size_t T = 64;
	for (size_t ii = 0; ii < rows; ii += T) {
		size_t i_max = std::min(ii + T, rows);
		for (size_t jj = 0; jj < cols; jj += T) {
			size_t j_max = std::min(jj + T, cols);
			for (size_t i = ii; i < i_max; ++i) {
				for (size_t j = jj; j < j_max; ++j) {
					result(j, i) = A(i, j);
				}
			}
		}
	}
//-------------------------------------------------------------------------------------------------------------------------------------------


	return result;
}
