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
int unroll = 8; //unroll factor, can be adjusted as needed
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < P; k += unroll) {
                for (int i1 = 0; i1 < unroll; i1++) {
                    C[i][j] += A[i][k + i1] * B[k + i1][j];
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
