#include <stdlib.h>

// define the matrix dimensions A is MxP, B is PxN, and C is MxN
#define M 512
#define N 512
#define P 512

// calculate C = AxB
void matmul(float **A, float **B, float **C) {
  float sum;
  int   i;
  int   j;
  int   k;
  int  B2 = 64;
  int  B1 = 8;
for(int i=0; i<N; i=i+B2) {
                for(int j=0; j<N; j=j+B2) {
                        for(int k=0; k<P; k=k+B2) {
                                for(int i_b=i; i_b<i+B2; i_b = i_b + B1) {
                                        for(int j_b=j; j_b<j+B2; j_b = j_b+B1) {
                                                for(int k_b=k; k_b<k+B2; k_b = k_b = k_b + B1) {
                                                        for(int i_b_b=i_b; i_b_b<i_b+B1; i_b_b++) {
                                                                for(int j_b_b=j_b; j_b_b<j_b+B1; j_b_b++ ) {
                                                                        for(int k_b_b=k_b; k_b_b<k_b+B1; k_b_b++) {
								C[i_b_b][j_b_b] = C[i_b_b][j_b_b] + (A[i_b_b][k_b_b] * B[k_b_b][j_b_b]);

			}
			}
			}
			}
			}
			}
			}
			}
			}
			}



// functi(n to allocate a matrix on the heap
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
  float** A;
  float** B;
  float** C;

  create_matrix(&A, M, P);
  create_matrix(&B, P, N);
  create_matrix(&C, M, N);

  // assume some initialization of A and B
  // think of this as a library where A and B are
  // inputs in row-major format, and C is an output
  // in row-major.

  matmul(A, B, C);

  return (0);
}
