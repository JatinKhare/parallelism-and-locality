#include <stdlib.h>
#include <xmmintrin.h>

// define the matrix dimensions A is MxP, B is PxN, and C is MxN
#define M 512
#define N 512
#define P 512
#define BLOCK_SIZE_I 256
#define BLOCK_SIZE_J 256
#define BLOCK_SIZE_K 256

// calculate C = AxB
void matmul(double **A, double  **B, double **C) {
    for (int i = 0; i < M; i += BLOCK_SIZE_I) {
        for (int j = 0; j < N; j += BLOCK_SIZE_J) {
            for (int l = 0; l < P; l += BLOCK_SIZE_K) {
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
void create_matrix(double*** A, int m, int n) {
  double **T = 0;
  int i;

  T = (double**)malloc( m*sizeof(double*));
  for ( i=0; i<m; i++ ) {
     T[i] = (double*)malloc(n*sizeof(double));
  }
  *A = T;
}

int main() {
  double** A;
  double** B;
  double** C;

  create_matrix(&A, M, P);
  for(int i = 0; i<M ;i++)
	  for(int j = 0; j<P ;j++)
		A[i][j] = i-j;
  create_matrix(&B, P, N);
  create_matrix(&C, M, N);

  // assume some initialization of A and B
  // think of this as a library where A and B are
  // inputs in row-major format, and C is an output
  // in row-major.

  matmul(A, B, C);

  return (0);
}
