#include <stdlib.h>
#include <xmmintrin.h>

// define the matrix dimensions A is MxP, B is PxN, and C is MxN
#define m 32
#define n 32
#define k 32
#define BLOCK_SIZE_I 16
#define BLOCK_SIZE_J 16
#define BLOCK_SIZE_K 16

// calculate C = AxB
void matmul(double **A, double  **B, double **C) {
    for (int i = 0; i < m; i += BLOCK_SIZE_I) {
        for (int j = 0; j < n; j += BLOCK_SIZE_J) {
            for (int l = 0; l < k; l += BLOCK_SIZE_K) {
                for (int ii = i; ii < i + BLOCK_SIZE_I; ii += BLOCK_SIZE_I/4) {
                    for (int jj = j; jj < j + BLOCK_SIZE_J; jj += BLOCK_SIZE_J/4) {
                        for (int ll = l; ll < l + BLOCK_SIZE_K; ll += BLOCK_SIZE_K/4) {
                            for (int iii = ii; iii < ii + BLOCK_SIZE_I/4; iii++) {
                                for (int jjj = jj; jjj < jj + BLOCK_SIZE_J/4; jjj++) {
                                    __builtin_prefetch(A[iii] + ll + BLOCK_SIZE_K, 0, 0);
                                    __builtin_prefetch(B[ll + BLOCK_SIZE_K] + jjj, 0, 0);
                                    double c = 0;
                                    for (int kkk = ll; kkk < ll + BLOCK_SIZE_K; kkk += 4) {
                                        __m128d a = _mm_loadu_pd(A[iii] + kkk);
                                        __m128d b = _mm_loadu_pd(B[kkk] + jjj);
                                        c += a[0]*b[0] + a[1]*b[1];
                                    }
                                    C[iii][jjj] += c;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// function to allocate a matrix on the heap
// creates an mXn matrix and returns the pointer.
//
// the matrices are in row-major order.
void create_matrix(float*** A, int m, int n) {
  float **T = 0;
  int i;

  T = (float**)malloc( m*sizeof(float*));
  for ( i=0; i<m; i++ ) {
     T[i] = (float*)malloc(n*sizeof(float));
  }
  *A = T;
}

int main() {
  double** A;
  double** B;
  double** C;

  create_matrix(&A, m, k);
  create_matrix(&B, k, n);
  create_matrix(&C, m, n);

  // assume some initialization of A and B
  // think of this as a library where A and B are
  // inputs in row-major format, and C is an output
  // in row-major.

  matmul(A, B, C);

  return (0);
}
