#include <stdlib.h>

// Usage: 
// ./matmul <dimension> <block_size_arg1> <block_size_arg2> <block_size_arg3>
//    Eg: ./matmul 4096 32 -2 -2 = cache oblivious with 32 as cut off
//        ./matmul 4096 32 -1 -1 = single level blocking (B1 = 32)
//        ./matmul 512  64 32 -1 = two level blocking (B2 = 64, B1 = 32)
//        ./matmul 512  64 32  8 = three level blocking (B3 = 64, B2 = 32, B1 = 8)  

// Dimensions: 32, 512, 4096
// block_size_arg1:
//      -1 - Original mat mul
//      >0 - Depending on the number of block sizes given and value of other args, this arg can be cutoff for oblivious, or blocking factor for L1, L2 or LLC
// block_size_arg2:
//	-1 - Original mat mul (or) if block_size_arg1>0, one level blocking
//      -2 - Oblivious case, ignore value
//      >0 - Depending on the number of block sizes given and value of other args, this arg can blocking factor L2 or L1 
// block_size_arg3:
//	-1 - Original mat mul (or) if block_size_arg1 and block_size_arg2>0, two level blocking
//      -2 - Oblivious case, ignore value
//      >0 - Depending on the number of block sizes given and value of other args, this arg can blocking factor LLC

int Dim, B1, B2, B3;
// calculate C = AxB


// recursion function for cache oblivious
void recur(float **A, float **B, float **C, int i0, int i1, int j0, int j1,
           int k0, int k1, int cutoff) {
  int i, j, k, di = i1 - i0, dj = j1 - j0, dk = k1 - k0;
  if (di >= dj && di >= dk && di > cutoff) {
    int im = (i0 + i1) / 2;
    recur(A, B, C, i0, im, j0, j1, k0, k1, cutoff);
    recur(A, B, C, im, i1, j0, j1, k0, k1, cutoff);
  } else if (dj >= dk && dj > cutoff) {
    int jm = (j0 + j1) / 2;
    recur(A, B, C, i0, i1, j0, jm, k0, k1, cutoff);
    recur(A, B, C, i0, i1, jm, j1, k0, k1, cutoff);
  } else if (dk > cutoff) {
    int km = (k0 + k1) / 2;
    recur(A, B, C, i0, i1, j0, j1, k0, km, cutoff);
    recur(A, B, C, i0, i1, j0, j1, km, k1, cutoff);
  } else {
    // cut off case
    for (i = i0; i < i1; ++i)
      for (j = j0; j < j1; ++j)
        for (k = k0; k < k1; ++k) C[i][j] += A[i][k] * B[k][j];
  }
}


// function to execute different types of mat mul 
void matmul(float **A, float **B, float **C) {
  float sum;
  int i;
  int j;
  int k;

  int original = ((B1 == -1) && (B2 == -1) && (B3 == -1));
  int level1 = ((B2 == -1) && (B3 == -1)) && (!original);
  int level2 = (B3 == -1) && (!level1 && !original);
  int level3 = (B3 != -1 && (B3 != -2)) && (~level1 && ~level2);
  int oblivious =
      (B2 == -2) && (!original) && (!level1) && (!level2) && (!level3);
  int my_case = (oblivious << 4) | (level3 << 3) | (level2 << 2) |
                (level1 << 1) | original;
  // printf("original = %d, level1 = %d, level2 = %d, level3 = %d, oblivious =
  // %d\n",original, level1, level2, level3, oblivious);
  switch (my_case) {
    case 1: {
     // Normal ijk case
      for (i = 0; i < Dim; i++) {
        // for each row of C
        for (j = 0; j < Dim; j++) {
          // for each column of C
          for (k = 0; k < Dim; k++) {
            // printf("i = %d, j = %d, k = %d\n", i, j, k);
            C[i][j] += A[i][k] * B[k][j];
            // dot product of row from A and column from B
          }
        }    
      }
      /*// Loop interchange
        // (Interchange j and k loops and run)
	for (i = 0; i < Dim; i++) {
        // for each row of C
        for (k = 0; k < Dim; k++) {
          // for each column of C
          for (j = 0; j < Dim; j++) {
            // printf("i = %d, j = %d, k = %d\n", i, j, k);
            C[i][j] += A[i][k] * B[k][j];
            // dot product of row from A and column from B
          }
        }
      */
      break;
    }
    case 2: {
      // Single level blocking
      for (int i = 0; i < Dim; i = i + B1) {
        for (int j = 0; j < Dim; j = j + B1) {
          for (int k = 0; k < Dim; k = k + B1) {
            for (int i_1 = i; i_1 < i + B1; i_1++) {
              for (int j_1 = j; j_1 < j + B1; j_1++) {
                sum = 0.0f;  // temporary value
                for (int k_1 = k; k_1 < k + B1; k_1++) {
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
    case 4: {
      // Two level blocking
      for (int i = 0; i < Dim; i = i + B1) {
        for (int j = 0; j < Dim; j = j + B1) {
          for (int k = 0; k < Dim; k = k + B1) {
            for (int i_1 = i; i_1 < i + B1; i_1 = i_1 + B2) {
              for (int j_1 = j; j_1 < j + B1; j_1 = j_1 + B2) {
                for (int k_1 = k; k_1 < k + B1; k_1 = k_1 + B2) {
                  for (int i_2 = i_1; i_2 < i_1 + B2; i_2++) {
                    for (int j_2 = j_1; j_2 < j_1 + B2; j_2++) {
                      sum = 0.0f;  // temporary value
                      for (int k_2 = k_1; k_2 < k_1 + B2; k_2++) {
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
      // printf("L2\n");
      break;
    }
    case 8: {
      // Three level blocking
      for (int i = 0; i < Dim; i = i + B1) {
        for (int j = 0; j < Dim; j = j + B1) {
          for (int k = 0; k < Dim; k = k + B1) {
            for (int i_1 = i; i_1 < i + B1; i_1 = i_1 + B2) {
              for (int j_1 = j; j_1 < j + B1; j_1 = j_1 + B2) {
                for (int k_1 = k; k_1 < k + B1; k_1 = k_1 + B2) {
                  for (int i_2 = i_1; i_2 < i_1 + B2; i_2 = i_2 + B3) {
                    for (int j_2 = j_1; j_2 < j_1 + B2; j_2 = j_2 + B3) {
                      for (int k_2 = k_1; k_2 < k_1 + B2; k_2 = k_2 + B3) {
                        for (int i_3 = i_2; i_3 < i_2 + B3; i_3++) {
                          for (int j_3 = j_2; j_3 < j_2 + B3; j_3++) {
                            // sum = 0.0f;
                            for (int k_3 = k_2; k_3 < k_2 + B3; k_3++) {
                              C[i_3][j_3] =
                                  C[i_3][j_3] + (A[i_3][k_3] * B[k_3][j_3]);
                            }
                            // C[i_3][j_3] = sum;
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
      // printf("L3\n");
      break;
    }
    case 16: {
      // cache oblivious
      recur(A, B, C, 0, Dim, 0, Dim, 0, Dim, B1);
      break;
    }

    default: {
      // printf("Oops :( Case\n");
      break;
    }
  }
}

// function to allocate a matrix on the heap
// creates an mXn matrix and returns the pointer.
//
// the matrices are in row-major order.
void create_matrix(float ***A, int m, int n) {
  float **T = 0;
  int i;

  T = (float **)malloc(m * sizeof(float *));
  for (i = 0; i < m; i++) {
    T[i] = (float *)malloc(n * sizeof(float));
  }
  *A = T;
}

int main(int argc, char *argv[]) {
  float **A;
  float **B;
  float **C;

  // printf("args[0] = %d, %d, %d, %d, %d\n",*argv[0], *argv[1], atoi(argv[2]),
  // *argv[3], *argv[4]);
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
