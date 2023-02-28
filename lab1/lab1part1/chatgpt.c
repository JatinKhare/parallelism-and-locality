#include <stdio.h>
#include <stdlib.h>

void cache_oblivious_matrix_multiplication(int *A, int *B, int *C, int n) {
if (n <= 64) {
// Base case: perform standard matrix multiplication using nested loops
for (int i = 0; i < n; i++) {
for (int j = 0; j < n; j++) {
for (int k = 0; k < n; k++) {
C[i * n + j] += A[i * n + k] * B[k * n + j];
}
}
}
} else {
// Divide matrices A and B into quadrants
int n2 = n / 2;
int *A11 = A;
int *A12 = A + n2;
int *A21 = A + n2 * n;
int *A22 = A + n2 * n + n2;
int *B11 = B;
int *B12 = B + n2;
int *B21 = B + n2 * n;
int *B22 = B + n2 * n + n2;
    // Allocate memory for temporary matrices
    int *C11 = (int *) malloc(n2 * n2 * sizeof(int));
    int *C12 = (int *) malloc(n2 * n2 * sizeof(int));
    int *C21 = (int *) malloc(n2 * n2 * sizeof(int));
    int *C22 = (int *) malloc(n2 * n2 * sizeof(int));
    int *M1 = (int *) malloc(n2 * n2 * sizeof(int));
    int *M2 = (int *) malloc(n2 * n2 * sizeof(int));
    int *M3 = (int *) malloc(n2 * n2 * sizeof(int));
    int *M4 = (int *) malloc(n2 * n2 * sizeof(int));
    int *M5 = (int *) malloc(n2 * n2 * sizeof(int));
    int *M6 = (int *) malloc(n2 * n2 * sizeof(int));
    int *M7 = (int *) malloc(n2 * n2 * sizeof(int));

    // Calculate temporary matrices
    cache_oblivious_matrix_multiplication(A11, B11, C11, n2);
    cache_oblivious_matrix_multiplication(A12, B21, C12, n2);
    cache_oblivious_matrix_multiplication(A11, B12, C21, n2);
    cache_oblivious_matrix_multiplication(A12, B22, C22, n2);
    cache_oblivious_matrix_multiplication(A21, B11, M1, n2);
    cache_oblivious_matrix_multiplication(A22, B21, M2, n2);
    cache_oblivious_matrix_multiplication(A21, B12, M3, n2);
    cache_oblivious_matrix_multiplication(A22, B22, M4, n2);

    // Combine temporary matrices to form final result

for (int i = 0; i < n2; i++) {
for (int j = 0; j < n2; j++) {
M5[i * n2 + j] = C11[i * n2 + j] + C12[i * n2 + j];
M6[i * n2 + j] = C21[i * n2 + j] + C22[i * n2 + j];
M7[i * n2 + j] = M1[i * n2 + j] + M2[i * n2 + j];
}
}

    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n2; j++) {
            C[i * n + j] = M5[i * n2 + j];
            C[i * n + j + n2] = M6[i * n2 + j];
            C[(i + n2) * n + j] = M7[i * n2 + j];
            C[(i + n2) * n + j + n2] = M4[i * n2 + j];
        }
    }

    // Free memory
    free(C11);
    free(C12);
    free(C21);
    free(C22);
    free(M1);
    free(M2);
    free(M3);
    free(M4);
    free(M5);
    free(M6);
    free(M7);
}
}

int main() {
// Example usage:
int A[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
int B[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
int C[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
cache_oblivious_matrix_multiplication(&A[0][0], &B[0][0], &C[0][0], 4);

// Print result
for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
        printf("%d ", C[i][j]);
    }
    printf("\n");
}
return 0;
}

/* Output:
90 100 110 120
202 228 254 280
314 356 398 440
426 484 542 600
*/



