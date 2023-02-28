#include <stdlib.h>

// define the matrix dimensions A is MxP, B is PxN, and C is MxN
/*#define M 512
#define N 512
#define P 512*/

int Dim, B1, B2, B3;
// calculate C = AxB


void matmul(float **A, float **B, float **C) {
  float sum;
  int   i;
  int   j;
  int   k;
	
	int original = ((B1==-1)&&(B2==-1)&&(B3==-1));
	int level1 = ((B2==-1)&&(B3==-1))&&(!original);
	int level2 = (B3==-1)&&(!level1 && !original);
	int level3 = (B3!=-1)&&(~level1 && ~level2);
	int my_case =  (level3 << 3) | (level2 << 2) | (level1 << 1) | original;
	printf("original = %d, level1 = %d, level2 = %d, level3 = %d\n",original, level1, level2, level3);
	switch(my_case){

					case 1:{

												 for (i=0; i<Dim; i++) {
																 // for each row of C
																 for (j=0; j<Dim; j++) {
																				 // for each column of C
																				 sum = 0.0f; // temporary value
																				 for (k=0; k<Dim; k++) {
																								 //printf("i = %d, j = %d, k = %d\n", i, j, k);
																								 sum += A[i][k]*B[k][j];
																								 // dot product of row from A and column from B
																				 }
																				 C[i][j] = sum;
																 }
												 }
												 break;
								 }
					case 2:{
												 for(int i=0; i<Dim; i=i+B1) {
																 for(int j=0; j<Dim; j=j+B1) {
																				 for(int k=0; k<Dim; k=k+B1) {
																								 for(int i_1=i; i_1<i+B1; i_1++) {
																												 for(int j_1=j; j_1<j+B1; j_1++) {
																																				 sum = 0.0f; // temporary value
																																 for(int k_1=k; k_1<k+B1; k_1++) {
																																				 sum += (A[i_1][k_1] * B[k_1][j_1]);
																																 }
																																				 C[i_1][j_1] = sum;
																												 }
																								 }
																				 }
																 }
												 }
												 break;
								 }
					case 4:{
				 for(int i=0; i<Dim; i=i+B1) {
								 for(int j=0; j<Dim; j=j+B1) {
												 for(int k=0; k<Dim; k=k+B1) {

																 for(int i_1=i; i_1<i+B1; i_1=i_1+B2) {
																				 for(int j_1=j; j_1<j+B1; j_1=j_1+B2 ) {
																								 for(int k_1=k; k_1<k+B1; k_1=k_1+B2) {

																												 for(int i_2=i_1; i_2<i_1+B2; i_2++) {
																																 for(int j_2=j_1; j_2<j_1+B2; j_2++) {
																																				 sum = 0.0f; // temporary value
																																				 for(int k_2=k_1; k_2<k_1+B2; k_2++) {
																																								 sum += (A[i_2][k_2] * B[k_2][j_2]);

																																				 }
																																								 C[i_2][j_2] = sum;
																																								 }
																																				 }
																																 }
																												 }
																								 }
																				 }
																 }
												 }
												 //printf("L2\n");
												 break;
								 }
					case 8:{
				 for(int i=0; i<Dim; i=i+B1) {
								 for(int j=0; j<Dim; j=j+B1) {
												 for(int k=0; k<Dim; k=k+B1) {

																 for(int i_1=i; i_1<i+B1; i_1=i_1+B2) {
																				 for(int j_1=j; j_1<j+B1; j_1=j_1+B2 ) {
																								 for(int k_1=k; k_1<k+B1; k_1=k_1+B2) {

																												 for(int i_2=i_1; i_2<i_1+B2; i_2=i_2+B3) {
																																 for(int j_2=j_1; j_2<j_1+B2; j_2=j_2+B3 ) {
																																				 for(int k_2=k_1; k_2<k_1+B2; k_2=k_2+B3) {

																																								 for(int i_3=i_2; i_3<i_2+B3; i_3++) {
																																												 for(int j_3=j_2; j_3<j_2+B3; j_3++) {
																																																 //sum = 0.0f;
																																																 for(int k_3=k_2; k_3<k_2+B3; k_3++) {
																																																				 C[i_3][j_3] = C[i_3][j_3] + (A[i_3][k_3] * B[k_3][j_3]);
																																																 }
																																																 //C[i_3][j_3] = sum;
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
				 }
				 //printf("L3\n");
				 break;
								 }
					case 16:
								 {
												 recur(A, B, C, 0, Dim, 0, Dim, 0, Dim, B1);
											  	break;

								 }

					default:{
													//printf("Oops :( Case\n");
													break;
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

int main(int argc, char *argv[]) {
				float** A;
				float** B;
				float** C;

				//printf("args[0] = %d, %d, %d, %d, %d\n",*argv[0], *argv[1], atoi(argv[2]), *argv[3], *argv[4]);
				Dim = atoi(argv[1]);
				B1 = atoi(argv[2]);
				B2 = atoi(argv[3]);
				B3 = atoi(argv[4]);
				create_matrix(&A, Dim, Dim);
				create_matrix(&B, Dim, Dim);
				create_matrix(&C, Dim, Dim);
				/*create_matrix(&A, M, P);
				create_matrix(&B, P, N);
				create_matrix(&C, M, N);*/

				// assume some initialization of A and B
				// think of this as a library where A and B are
				// inputs in row-major format, and C is an output
				// in row-major.

				matmul(A, B, C);

				return (0);
}
