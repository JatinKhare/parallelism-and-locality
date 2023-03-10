#include <stdlib.h>

// define the matrix dimensions A is MxP, B is PxN, and C is MxN
#define M 512 
#define N 512
#define P 512
#define Dim 512
#define B1 16

// calculate C = AxB
void matmul(float **A, float **B, float **C) {
				float sum;
				int   i;
				int   j;
				int   k;

				for (i=0; i<M; i++) {
								// for each row of C
								//
								/*for (j=0; j<N; j++) {
												// for each column of C
												sum = 0.0f; // temporary value
												for (k=0; k<P; k++) {
																// dot product of row from A and column from B
																sum += A[i][k]*B[k][j];
												}
												C[i][j] = sum;
								}*/
								for (k=0; k<P; j++) {
												// for each column of C
												sum = 0.0f; // temporary value
												for (j=0; j<N; j++) {
																// dot product of row from A and column from B
																sum += A[i][j]*B[j][k];
												}
												C[i][k] = sum;
								}
				}

				/*for(int i=0; i<Dim; i=i+B1) {
								for(int j=0; j<Dim; j=j+B1) {
												for(int k=0; k<Dim; k=k+B1) {
																for(int i_1=i; i_1<i+B1; i_1++) {
																				for(int j_1=j; j_1<j+B1; j_1++) {
																								for(int k_1=k; k_1<k+B1; k_1++) {
																												C[i_1][j_1] = C[i_1][j_1] + (A[i_1][k_1] * B[k_1][j_1]);

																								}
																				}
																}
												}
								}
				}*/
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
