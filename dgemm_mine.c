const char* dgemm_desc = "My awesome dgemm.";

/* for the L2 cache */
#define B1_X  ((const int) 8)
#define B1_ID ((const int) 8) /* inner dimension */
#define B1_Y  ((const int) 8)

/* for the L1 cache */
#define B2_X  ((const int) 8)
#define B2_ID ((const int) 8) /* inner dimension */
#define B2_Y  ((const int) 8)

/* for the registers */
#define B3_X  ((const int) 4)
#define B3_ID ((const int) 8) /* inner dimension */
#define B3_Y  ((const int) 4)

/* for using memset */
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void basic_dgemm(const int M, const int P, const int N, const double * restrict A, 
		 const double * restrict B, double * restrict C)
{
  /* perform calculations on smallest block */
  /* A is row-major, B is column-major, C is column-major */
  int i, j, k;
#pragma simd
  for (j = 0; j < B3_Y; ++j) {
#pragma simd
    for (i = 0; i < B3_X; ++i) {
      double cij = C[i+j*M];
#pragma simd
      for (k = 0; k < B3_ID; ++k) {
	cij += A[k+i*P] * B[k+j*P];
      }
      C[i+j*M] = cij;
    }
  }
}
    
void make_block2(const int M, const int P, const int N, const double * restrict A, 
		 const double * restrict B, double * restrict C)
{
  /* create third level of blocks */
  /* send to basic dgemm solver */
  for (int i = 0; i < B2_X ; i += B3_X) {
    for (int j = 0; j < B2_Y ; j += B3_Y) {
      for (int k = 0; k < B2_ID; k +=B3_ID) { 
	basic_dgemm(M, P, N, A + k + i*P, B + k + j*P, C + i + j*M);
      }
    }
  }
}

void make_block1(const int M, const int P, const int N, const double * restrict A, 
		 const double * restrict B, double * restrict C)
{
  /* create second level of blocks */
  for (int i = 0; i < B1_X ; i +=B2_X) {
    for (int j = 0; j < B1_Y ; j +=B2_Y) {
      for (int k = 0; k < B1_ID; k +=B2_ID) { 
	make_block2(M, P, N, A + k + i*P, B + k + j*P, C + i + j*M);
      }
    }
  }
}

void square_dgemm(const int lda, const double * restrict AA, const double * restrict BB, double * restrict CC)
{
  /* determine how much larger the padded matrices need to be */
  /* new matrices are M x P, P x N, M x N */
  int M = (lda/B1_X  + (lda % B1_X ? 1 : 0))*B1_X ;
  int P = (lda/B1_ID + (lda % B1_ID? 1 : 0))*B1_ID;
  int N = (lda/B1_Y  + (lda % B1_Y ? 1 : 0))*B1_Y ;
  /* need to fix this for the case when the modulus is 0 */

  /* pad A and B with zeros according to block sizes, and pass new A and B to make_block1 */
  /* create new arrays by allocating memory */
  double * restrict A = (double*) malloc(M*P*sizeof(double));
  double * restrict B = (double*) malloc(P*N*sizeof(double));
  double * restrict C = (double*) malloc(M*N*sizeof(double));
  /* set all values in new arrays to be 0 */
  memset(A, 0, M*P*sizeof(double));
  memset(B, 0, P*N*sizeof(double));
  memset(C, 0, M*N*sizeof(double));

  /* print values of interest */
  /* printf("lda is %d\n", lda);
  printf("M is %d\n", M);
  printf("P is %d\n", P);
  printf("N is %d\n", N);
  printf("B1_X  is %d\n", B1_X );
  printf("B1_ID is %d\n", B1_ID);
  printf("B1_Y  is %d\n", B1_Y ); */

  /* for troubleshooting */
  /* printf("matrix AA\n");
  for (int i = 0; i < lda; ++i) {
    for (int j = 0; j < lda; ++j) {
      printf("%f ", AA[i+j*lda]);
    }
    printf("\n");
  }  */

  /* copy the values of original matrices to padded matrices */
  for (int i = 0; i < lda; ++i) {
    for (int j = 0; j < lda; ++j) {
      B[i+j*P] = BB[i+j*lda];
      A[j+i*P] = AA[i+j*lda]; /* make A row-major*/
    }
  }
 
  /* for troubleshooting */
  /* printf("%f\n", A[25+25*P]);
     printf("%f\n", AA[25+25*lda]); */
  /*printf("matrix AA - matrix A\n");
  for (int i = 0; i < lda; ++i) {
    for (int j = 0; j < lda; ++j) {
      double AmA =  AA[i+j*lda]-A[j+i*P];
      printf("%f ", AmA);
    }
    printf("\n");
    }  */

  /* create first level of blocks */
  for (int i = 0; i < M; i += B1_X) {
    for (int j = 0; j < N; j += B1_Y) {
      for (int k = 0; k < P; k +=B1_ID) { 
	make_block1(M, P, N, A + k + i*P, B + k + j*P, C + i + j*M);
      }
    }
  }

  free(A);
  free(B);

  /* put data in padded C back into original CC */
  for (int i = 0; i < lda; ++i) {
    for (int j = 0; j < lda; ++j) {
      CC[i+j*lda] = C[i+j*M];
    }
  }

  free(C);
}
