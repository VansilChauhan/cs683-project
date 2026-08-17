/*******************************************************************
 * Author: <Name1>, <Name2>
 * Date: <Date>
 * File: mat_mul.c
 * Description: This file contains implementations of matrix multiplication
 *			    algorithms using various optimization techniques.
 *******************************************************************/

// PA 1: Matrix Multiplication

// includes
#include <stdio.h>
#include <stdlib.h>         // for malloc, free, atoi
#include <time.h>           // for time()
#include <chrono>	        // for timing
#include <xmmintrin.h> 		// for SSE
#include <immintrin.h>		// for AVX

#include "helper.h"			// for helper functions

// defines
// NOTE: you can change this value as per your requirement
#define TILE_SIZE	100		// size of the tile for blocking

/**
 * @brief 		Performs matrix multiplication of two matrices.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void naive_mat_mul(double *A, double *B, double *C, int size) {

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

/**
 * @brief 		Task 1A: Performs matrix multiplication of two matrices using loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void loop_opt_mat_mul(double *A, double *B, double *C, int size){
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            double a_val = A[i * size + k];
            int j;
            for (j = 0; j <= size-32; j+=32) {
                C[i * size + j] += a_val * B[k * size + j];
                C[i * size + j + 1] += a_val * B[k * size + j + 1];
                C[i * size + j + 2] += a_val * B[k * size + j + 2];
                C[i * size + j + 3] += a_val * B[k * size + j + 3];
                C[i * size + j + 4] += a_val * B[k * size + j + 4];
                C[i * size + j + 5] += a_val * B[k * size + j + 5];
                C[i * size + j + 6] += a_val * B[k * size + j + 6];
                C[i * size + j + 7] += a_val * B[k * size + j + 7];
                C[i * size + j + 8] += a_val * B[k * size + j + 8];
                C[i * size + j + 9] += a_val * B[k * size + j + 9];
                C[i * size + j + 10] += a_val * B[k * size + j + 10];
                C[i * size + j + 11] += a_val * B[k * size + j + 11];
                C[i * size + j + 12] += a_val * B[k * size + j + 12];
                C[i * size + j + 13] += a_val * B[k * size + j + 13];
                C[i * size + j + 14] += a_val * B[k * size + j + 14];
                C[i * size + j + 15] += a_val * B[k * size + j + 15];
                C[i * size + j + 16] += a_val * B[k * size + j + 16];
                C[i * size + j + 17] += a_val * B[k * size + j + 17];
                C[i * size + j + 18] += a_val * B[k * size + j + 18];
                C[i * size + j + 19] += a_val * B[k * size + j + 19];
                C[i * size + j + 20] += a_val * B[k * size + j + 20];
                C[i * size + j + 21] += a_val * B[k * size + j + 21];
                C[i * size + j + 22] += a_val * B[k * size + j + 22];
                C[i * size + j + 23] += a_val * B[k * size + j + 23];
                C[i * size + j + 24] += a_val * B[k * size + j + 24];
                C[i * size + j + 25] += a_val * B[k * size + j + 25];
                C[i * size + j + 26] += a_val * B[k * size + j + 26];
                C[i * size + j + 27] += a_val * B[k * size + j + 27];
                C[i * size + j + 28] += a_val * B[k * size + j + 28];
                C[i * size + j + 29] += a_val * B[k * size + j + 29];
                C[i * size + j + 30] += a_val * B[k * size + j + 30];
                C[i * size + j + 31] += a_val * B[k * size + j + 31];
            }
            for(; j < size; j++){
                C[i * size + j] += a_val * B[k * size + j];
            }
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------

}


/**
 * @brief 		Task 1B: Performs matrix multiplication of two matrices using tiling.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * 				For example, if the dimension is 1024, then the tile size can be 32, 64, 128, etc.
 * 				You can assume that the matrices are square matrices.
*/
void tile_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for (int it = 0; it < size; it+=tile_size){
        for (int jt = 0; jt < size; jt+=tile_size){
            for (int kt = 0; kt < size; kt += tile_size){
                for (int i = it; i < it+tile_size && i < size; i++) {
                    for (int j = jt; j < jt+tile_size && j < size; j++) {
                        double temp = 0;
                        for (int k = kt; k < kt+tile_size && k < size; k++) {
                            temp += A[i * size + k] * B[k * size + j];
                        }
                        C[i * size + j] += temp;
                    }
                }			
            }
        }
    }    
//-------------------------------------------------------------------------------------------------------------------------------------------
    
}

/**
 * @brief 		Task 1C: Performs matrix multiplication of two matrices using SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/
void simd_mat_mul(double *A, double *B, double *C, int size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    const int V = 8; 

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j <= size - V; j += V) {
            __m512d cvec = _mm512_loadu_pd(&C[i * size + j]);

            for (int k = 0; k < size; ++k) {
                __m512d avec = _mm512_set1_pd(A[i * size + k]);

                __m512d bvec = _mm512_loadu_pd(&B[k * size + j]);

                cvec = _mm512_fmadd_pd(avec, bvec, cvec);
            }

            _mm512_storeu_pd(&C[i * size + j], cvec);
        }

        for (int j = (size / V) * V; j < size; ++j) {
            double sum = C[i * size + j];
            for (int k = 0; k < size; ++k) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------
}

/**
 * @brief 		Task 1D: Performs matrix multiplication of two matrices using combination of tiling/SIMD/loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * @note 		You can assume that the matrices are square matrices.
*/
void combination_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    
    // loop reordering and unrolling
	// for (int i = 0; i < size; i++) {
    //     for (int k = 0; k < size; k++) {
	// 		double a_val = A[i * size + k];
    //         int j;
    //         for (j = 0; j <= size-32; j+=32) {
	// 			C[i * size + j] += a_val * B[k * size + j];
	// 			C[i * size + j + 1] += a_val * B[k * size + j + 1];
	// 			C[i * size + j + 2] += a_val * B[k * size + j + 2];
	// 			C[i * size + j + 3] += a_val * B[k * size + j + 3];
	// 			C[i * size + j + 4] += a_val * B[k * size + j + 4];
	// 			C[i * size + j + 5] += a_val * B[k * size + j + 5];
	// 			C[i * size + j + 6] += a_val * B[k * size + j + 6];
	// 			C[i * size + j + 7] += a_val * B[k * size + j + 7];
	// 			C[i * size + j + 8] += a_val * B[k * size + j + 8];
	// 			C[i * size + j + 9] += a_val * B[k * size + j + 9];
	// 			C[i * size + j + 10] += a_val * B[k * size + j + 10];
	// 			C[i * size + j + 11] += a_val * B[k * size + j + 11];
	// 			C[i * size + j + 12] += a_val * B[k * size + j + 12];
	// 			C[i * size + j + 13] += a_val * B[k * size + j + 13];
	// 			C[i * size + j + 14] += a_val * B[k * size + j + 14];
	// 			C[i * size + j + 15] += a_val * B[k * size + j + 15];
	// 			C[i * size + j + 16] += a_val * B[k * size + j + 16];
	// 			C[i * size + j + 17] += a_val * B[k * size + j + 17];
	// 			C[i * size + j + 18] += a_val * B[k * size + j + 18];
	// 			C[i * size + j + 19] += a_val * B[k * size + j + 19];
	// 			C[i * size + j + 20] += a_val * B[k * size + j + 20];
	// 			C[i * size + j + 21] += a_val * B[k * size + j + 21];
	// 			C[i * size + j + 22] += a_val * B[k * size + j + 22];
	// 			C[i * size + j + 23] += a_val * B[k * size + j + 23];
	// 			C[i * size + j + 24] += a_val * B[k * size + j + 24];
	// 			C[i * size + j + 25] += a_val * B[k * size + j + 25];
	// 			C[i * size + j + 26] += a_val * B[k * size + j + 26];
	// 			C[i * size + j + 27] += a_val * B[k * size + j + 27];
	// 			C[i * size + j + 28] += a_val * B[k * size + j + 28];
	// 			C[i * size + j + 29] += a_val * B[k * size + j + 29];
	// 			C[i * size + j + 30] += a_val * B[k * size + j + 30];
	// 			C[i * size + j + 31] += a_val * B[k * size + j + 31];
	// 		}
    //         for(; j < size; j++){
    //             C[i * size + j] += a_val * B[k * size + j];
    //         }
	// 	}
	// }

	// simd and loop reordering
    // const int V = 8; 

    // for (int i = 0; i < size; ++i) {
    //     for (int k = 0; k < size; ++k) {
    //         __m512d avec = _mm512_set1_pd(A[i * size + k]);

    //         int j = 0;
    //         for (; j <= size - V; j += V) {
    //             __m512d cvec = _mm512_loadu_pd(&C[i * size + j]);
    //             __m512d bvec = _mm512_loadu_pd(&B[k * size + j]);

    //             cvec = _mm512_fmadd_pd(avec, bvec, cvec);
    //             _mm512_storeu_pd(&C[i * size + j], cvec);
    //         }

    //         for (; j < size; ++j) {
    //             C[i * size + j] += A[i * size + k] * B[k * size + j];
    //         }
    //     }
    // }

	// tiling + reordering
    // for (int it = 0; it < size; it += tile_size) {
    //     for (int kt = 0; kt < size; kt += tile_size) {
    //         for (int jt = 0; jt < size; jt += tile_size) {

    //             for (int i = it; i < it + tile_size && i < size; i++) {
    //                 for (int k = kt; k < kt + tile_size && k < size; k++) {
    //                     double a_val = A[i * size + k]; // reuse A[i,k]
    //                     for (int j = jt; j < jt + tile_size && j < size; j++) {
    //                         C[i * size + j] += a_val * B[k * size + j];
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

	//tiling + simd
	// const int V = 8;

    // for (int it = 0; it < size; it += tile_size) {
    //     for (int jt = 0; jt < size; jt += tile_size) {
    //         for (int kt = 0; kt < size; kt += tile_size) {

    //             for (int i = it; i < it + tile_size && i < size; i++) {
    //                 for (int j = jt; j < jt + tile_size && j < size; j++) {
    //                     __m512d sum_vec = _mm512_setzero_pd();
    //                     int k = kt;

    //                     for (; k <= kt + tile_size - V && k + V <= size; k += V) {
    //                         __m512d avec = _mm512_loadu_pd(&A[i * size + k]);
    //                         __m512d bvec = _mm512_loadu_pd(&B[k * size + j]);
    //                         sum_vec = _mm512_fmadd_pd(avec, bvec, sum_vec);
    //                     }

    //                     double temp[8];
    //                     _mm512_storeu_pd(temp, sum_vec);
    //                     double sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

    //                     for (; k < kt + tile_size && k < size; k++) {
    //                         sum += A[i * size + k] * B[k * size + j];
    //                     }

    //                     C[i * size + j] += sum;
    //                 }
    //             }
    //         }
    //     }
    // }

	//tiling + simd + loop_reordering
	const int V = 8;

    for (int it = 0; it < size; it += tile_size) {
        for (int kt = 0; kt < size; kt += tile_size) {
            for (int jt = 0; jt < size; jt += tile_size) {

                for (int i = it; i < it + tile_size && i < size; i++) {
                    for (int k = kt; k < kt + tile_size && k < size; k++) {
                        __m512d avec = _mm512_set1_pd(A[i * size + k]);

                        int j = jt;
                        for (; j <= jt + tile_size - V && j + V <= size; j += V) {
                            __m512d cvec = _mm512_loadu_pd(&C[i * size + j]);
                            __m512d bvec = _mm512_loadu_pd(&B[k * size + j]);
                            cvec = _mm512_fmadd_pd(avec, bvec, cvec);
                            _mm512_storeu_pd(&C[i * size + j], cvec);
                        }

                        for (; j < jt + tile_size && j < size; j++) {
                            C[i * size + j] += A[i * size + k] * B[k * size + j];
                        }
                    }
                }
            }
        }
    }

//-------------------------------------------------------------------------------------------------------------------------------------------
    
}

// NOTE: DO NOT CHANGE ANYTHING BELOW THIS LINE
/**
 * @brief 		Main function
 * @param 		argc 		number of command line arguments
 * @param 		argv 		array of command line arguments
 * @return 		0 on success
 * @note 		DO NOT CHANGE THIS FUNCTION
 * 				DO NOT ADD OR REMOVE ANY COMMAND LINE ARGUMENTS
*/
int main(int argc, char **argv) {

    if ( argc <= 1 ) {
        printf("Usage: %s <matrix_dimension>\n", argv[0]);
        return 0;
    }

    else {
        int size = atoi(argv[1]);

        double *A = (double *)malloc(size * size * sizeof(double));
        double *B = (double *)malloc(size * size * sizeof(double));
        double *C = (double *)calloc(size * size, sizeof(double));

        // initialize random seed
        srand(time(NULL));

        // initialize matrices A and B with random values
        initialize_matrix(A, size, size);
        initialize_matrix(B, size, size);

        // perform normal matrix multiplication
        auto start = std::chrono::high_resolution_clock::now();
        naive_mat_mul(A, B, C, size);
        auto end = std::chrono::high_resolution_clock::now();
        auto time_naive_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Normal matrix multiplication took %ld ms to execute \n\n", time_naive_mat_mul);

    #ifdef OPTIMIZE_LOOP_OPT
        // Task 1a: perform matrix multiplication with loop optimization

        // initialize result matrix to 0
        initialize_result_matrix(C, size, size);

        start = std::chrono::high_resolution_clock::now();
        loop_opt_mat_mul(A, B, C, size);
        end = std::chrono::high_resolution_clock::now();
        auto time_loop_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Loop optimized matrix multiplication took %ld ms to execute \n", time_loop_mat_mul);
        printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_loop_mat_mul);
    #endif

    #ifdef OPTIMIZE_TILING
        // Task 1b: perform matrix multiplication with tiling

        // initialize result matrix to 0
        initialize_result_matrix(C, size, size);

        start = std::chrono::high_resolution_clock::now();
        tile_mat_mul(A, B, C, size, TILE_SIZE);
        end = std::chrono::high_resolution_clock::now();
        auto time_tiling_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Tiling matrix multiplication took %ld ms to execute \n", time_tiling_mat_mul);
        printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_tiling_mat_mul);
    #endif

    #ifdef OPTIMIZE_SIMD
        // Task 1c: perform matrix multiplication with SIMD instructions 

        // initialize result matrix to 0
        initialize_result_matrix(C, size, size);

        start = std::chrono::high_resolution_clock::now();
        simd_mat_mul(A, B, C, size);
        end = std::chrono::high_resolution_clock::now();
        auto time_simd_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        printf("SIMD matrix multiplication took %ld ms to execute \n", time_simd_mat_mul);
        printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_simd_mat_mul);
    #endif

    #ifdef OPTIMIZE_COMBINED
        // Task 1d: perform matrix multiplication with combination of tiling, SIMD and loop optimization

        // initialize result matrix to 0
        initialize_result_matrix(C, size, size);

        start = std::chrono::high_resolution_clock::now();
        combination_mat_mul(A, B, C, size, TILE_SIZE);
        end = std::chrono::high_resolution_clock::now();
        auto time_combination = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Combined optimization matrix multiplication took %ld ms to execute \n", time_combination);
        printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_combination);
    #endif

        // free allocated memory
        free(A);
        free(B);
        free(C);

        return 0;
    }
}
