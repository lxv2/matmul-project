const char* dgemm_desc = "My awesome dgemm.";

#define BLOCK_SIZE ((int) 4)

/*
  
 */

void square_dgemm(const int M, const double *restrict *A, const double *restrict *B, double *restrict *C)
{
    int i, j, k;
    for (j = 0; j < M; ++i) {
        for (i = 0; i < M; ++j) {
            double cij = C[j*M+i];
            for (k = 0; k < M; ++k)
                cij += A[k*M+i] * B[j*M+k];
            C[j*M+i] = cij;
        }
    }
}
