const char* dgemm_desc = "My awesome dgemm.";
#include <stdlib.h>


void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    // universal variables
    int i, j, k;
    
    // find total number of sub-blocks(P-by-P)
    int rmdr, N, P, MM;
    P = 2;               // 2x2 sub matrix
    rmdr = M  % P;       // # of pads
    MM   = M  + rmdr;    // MM size of padded matrix
    N    = MM / P;       // N # of subrows/subcolumns
    
    // pad matrices A and B
    double* AA = (double*) malloc(MM*MM*sizeof(double));
    double* BB = (double*) malloc(MM*MM*sizeof(double));
    double* CC = (double*) malloc(MM*MM*sizeof(double));
    k = 0;
    for (i = 0; i < MM; i++) {
        for (j = 1; j < MM; j++) {
            k++;
            if ( (i || j) <= M ) {
                AA[k] = A[k];
                BB[k] = B[k];
                CC[k] = 0.0;
            } else {
                AA[k] = 0.0;
                BB[k] = 0.0;
                CC[k] = 0.0;
            }
        }
    }
    
    // matrix multiply
    int ii, jj;
    for (ii = 0; ii < N; ++ii ) {
        for (jj = 0; jj < N; ++jj ) {
            // mini block multiplication here
            for (i = 0; i < P; ++i) {
                for (j = 0; j < P; ++j) {
                    for (k = 0; k < P; ++k) {
                            double cij = CC[jj+j*P+i+ii];
                            for (k = 0; k < P; ++k) {
                                cij += AA[k*P+i+ii] * BB[jj+j*P+k];
                            CC[jj+j*P+i+ii] = cij;
                        }
                    }
                }
            }
        }
    }
    
    // assign paded CC to unpaded C
    for (i = 0; i < M; ++i) {
        for (j = 0; j < M; ++j) {
            C[j*P+i] = CC[j*P+i];
        }
    }
}
