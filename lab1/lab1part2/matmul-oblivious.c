#include <stdio.h>
#include <stdlib.h>
#define N 4096
#define CUTOFF 2048
// function to allocate a matrix on the heap
// creates an mXn matrix and returns the pointer.
//
// the matrices are in row-major order.
void create_matrix(float ** * A, int m, int n) {
  float ** T = 0;
  int i;

  T = (float ** ) malloc(m * sizeof(float * ));
  for (i = 0; i < m; i++) {
    T[i] = (float * ) malloc(n * sizeof(float));
  }
  * A = T;
}

void recur(float ** A, float ** B, float ** C, int i0, int i1, int j0, int j1, int k0, int k1, int cutoff) {
  int i, j, k, di = i1 - i0, dj = j1 - j0, dk = k1 - k0;
  if (di >= dj && di >= dk && di > cutoff) {
    int im = (i0 + i1) / 2;
    recur(A, B, C, i0, im, j0, j1, k0, k1, cutoff);
    recur(A, B, C, im, i1, j0, j1, k0, k1, cutoff);
    //printf("Dividing\n");
  } else if (dj >= dk && dj > cutoff) {
    int jm = (j0 + j1) / 2;
    recur(A, B, C, i0, i1, j0, jm, k0, k1, cutoff);
    recur(A, B, C, i0, i1, jm, j1, k0, k1, cutoff);
    //printf("Dividing\n");
  } else if (dk > cutoff) {
    int km = (k0 + k1) / 2;
    recur(A, B, C, i0, i1, j0, j1, k0, km, cutoff);
    recur(A, B, C, i0, i1, j0, j1, km, k1, cutoff);
    //printf("Dividing\n");
  } else {
    //printf("Base Case\n");
    for (i = i0; i < i1; ++i)
      for (j = j0; j < j1; ++j)
        for (k = k0; k < k1; ++k)
          C[i][j] += A[i][k] * B[k][j];
  }
}

int main(int argc, char * argv[]) {
  float ** A;
  float ** B;
  float ** C;
  create_matrix( & A, N, N);
  create_matrix( & B, N, N);
  create_matrix( & C, N, N);

  // assume some initialization of A and B
  // think of this as a library where A and B are
  // inputs in row-major format, and C is an output
  // in row-major.

  //matrix_mult(A,B,C,N);
  recur(A, B, C, 0, N, 0, N, 0, N, CUTOFF);
  free(A);
  free(B);
  free(C);

  return (0);
}
