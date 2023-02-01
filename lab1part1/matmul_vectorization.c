#include <stdlib.h>
#include <xmmintrin.h>
#include <immintrin.h>

// define the matrix dimensions A is MxP, B is PxN, and C is MxN
#define M 64
#define N 64
#define P 64

void matmul(double** A, double** B, double** C) {
for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
           // __m256d c_ij = _mm256_setzero_pd();  // Initialize c_ij to zero
            __m256d c_ij;
            for (int k = 0; k < P; k++) {
                __m256d a_ik = _mm256_loadu_pd(A + i * P + k * 4);  // Load 4 elements of A[i][k] into a_ik
                __m256d b_kj = _mm256_loadu_pd(B + k * N + j * 4);  // Load 4 elements of B[k][j] into b_kj
                c_ij = _mm256_add_pd(c_ij, _mm256_mul_pd(a_ik, b_kj));  // Multiply and add to c_ij
            }
            _mm256_storeu_pd(C + i * N + j * 4, c_ij);  // Store c_ij in C[i][j]
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
  for(int i = 0; i<P ;i++)
	  for(int j = 0; j<N ;j++)
		B[i][j] = i-j;
  create_matrix(&C, M, N);

  // assume some initialization of A and B
  // think of this as a library where A and B are
  // inputs in row-major format, and C is an output
  // in row-major.

  matmul(A, B, C);

  for(int i = 0; i<M ;i++)
	  for(int j = 0; j<N ;j++)
		printf("C[%d][%d] = %d", i, j, C[i][j]);
  return (0);
}
