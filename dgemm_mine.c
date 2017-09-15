const char* dgemm_desc = "My awesome dgemm.";

/* for the L2 cache */
#define B1_X  ((const int) 16)
#define B1_ID ((const int) 16) /* inner dimension */
#define B1_Y  ((const int) 16)

/* for the L1 cache */
#define B2_X  ((const int) 16)
#define B2_ID ((const int) 16) /* inner dimension */
#define B2_Y  ((const int) 16)

/* for the registers */
#define B3_X  ((const int) 8)
#define B3_ID ((const int) 16) /* inner dimension */
#define B3_Y  ((const int) 16)

/* for using memset */
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

inline void basic_dgemm(const double Ablock3[B3_X*B3_ID], const double Bblock3[B3_Y*B3_ID],
			double Cblock3[B3_X*B3_Y])
{
  /* perform calculations on smallest block */
  /* A is row-major, B is column-major, C is column-major */
  int i, j, k;
#pragma simd
  for (j = 0; j < B3_Y; ++j) {
#pragma simd
    for (i = 0; i < B3_X; ++i) {
      double cij = Cblock3[i+j*B3_ID];
#pragma simd
      for (k = 0; k < B3_ID; ++k) {
	cij += Ablock3[k+i*B3_ID] * Bblock3[k+j*B3_ID];
      }
      Cblock3[i+j*B3_ID] = cij;
    }
  }
}
    
inline void make_block2(const double Ablock2[B2_X*B2_ID], const double Bblock2[B2_Y*B2_ID],
			double Cblock2[B2_X*B2_Y])
{
  /* create third level of blocks */
  double Ablock3[B3_X*B3_ID] = {0};
  double Bblock3[B3_Y*B3_ID] = {0};
  double Cblock3[B3_X*B3_Y ] = {0};

  /* send to basic dgemm solver */
  for (int i = 0; i < B2_X ; i += B3_X) {
    for (int j = 0; j < B2_Y ; j += B3_Y) {
      for (int k = 0; k < B2_ID; k +=B3_ID) { 
	// copy inputs to block series 3
	for (int kk = 0; kk < B3_ID; ++kk) {	
	  for (int ii = 0; ii < B3_X; ++ii) {
	    Ablock3[kk+ii*B3_ID] = Ablock2[(k+kk)+(i+ii)*B2_ID];
	  }
	  for (int jj = 0; jj < B3_Y; ++jj) {
	    Bblock3[kk+jj*B3_ID] = Bblock2[(k+kk)+(j+jj)*B2_ID];
	  }
	}
	printf("matrix Ablock3\n");
	for (int iii = 0; iii < B3_X; ++iii) {
	  for (int jjj = 0; jjj < B3_ID; ++jjj) {
	    double AmA =  Ablock3[iii+jjj*B3_X];
	    printf("%f ", AmA);
	  }
	  printf("\n");
	}
	basic_dgemm(Ablock3, Bblock3, Cblock3);
	// copy outputs from block series 3
	for (int ii = 0; ii < B3_X; ++ii) {
	  for (int jj = 0; jj < B3_Y; ++jj) {
	    Cblock2[(i+ii)+(j+jj)*B2_X] += Cblock3[ii+jj*B3_X];
	  }
	}
      }
    }
  }
}

inline void make_block1(const double Ablock1[B1_X*B1_ID], const double Bblock1[B1_Y*B1_ID],
			double Cblock1[B1_X*B1_Y])
{
  /* create second level of blocks */
  double Ablock2[B2_X*B2_ID] = {0};
  double Bblock2[B2_Y*B2_ID] = {0};
  double Cblock2[B2_X*B2_Y ] = {0};

  int ct = 0;

  for (int i = 0; i < B1_X ; i +=B2_X) {
    for (int j = 0; j < B1_Y ; j +=B2_Y) {
    ct += 1;
      for (int k = 0; k < B1_ID; k +=B2_ID) {
	// copy to inputs to block series 2
	for (int kk = 0; kk < B2_ID; ++kk) {
	  for (int ii = 0; ii < B2_X; ++ii) {
	    Ablock2[kk+ii*B2_ID] = Ablock1[(k+kk)+(i+ii)*B1_ID];
	  }
	  for (int jj = 0; jj < B2_Y; ++jj) {
	    Bblock2[kk+jj*B2_ID] = Bblock1[(k+kk)+(j+jj)*B1_ID];
	  }
	}
	//printf("matrix Ablock2\n");
	for (int iii = 0; iii < B1_ID; ++iii) {
	  for (int jjj = 0; jjj < B1_ID; ++jjj) {
	    double AmA =  Ablock2[jjj+iii*B1_ID];
	    //printf("%f ", AmA);
	  }
	  //printf("\n");
	}
	make_block2(Ablock2, Bblock2, Cblock2);
	// copy outputs from block series 2
	for (int ii = 0; ii < B2_X; ++ii) {
	  for (int jj = 0; jj < B2_Y; ++jj) {
	    Cblock1[(i+ii)+(j+jj)*B1_X] += Cblock2[ii+jj*B2_X];
	  }
	}
      }
    }
    //printf("make_block1 = %d\n",ct);

  }
}
    
void square_dgemm(const int lda, const double * restrict A, const double * restrict B, double * restrict C)
{
  /* determine how much larger the padded matrices need to be */
  /* new matrices are M x P, P x N, M x N */
  int M = (lda/B1_X  + (lda % B1_X ? 1 : 0))*B1_X ;
  int P = (lda/B1_ID + (lda % B1_ID? 1 : 0))*B1_ID;
  int N = (lda/B1_Y  + (lda % B1_Y ? 1 : 0))*B1_Y ;
  /* need to fix this for the case when the modulus is 0 */

  /* create blocks for each level */
  double Ablock1[B1_X*B1_ID] = {0};
  double Bblock1[B1_Y*B1_ID] = {0};
  double Cblock1[B1_X*B1_Y ] = {0};

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

  /* for troubleshooting */
  /* printf("%f\n", A[25+25*P]);
     printf("%f\n", AA[25+25*lda]); */
  //printf("matrix A\n");
  for (int i = 0; i < lda; ++i) {
    for (int j = 0; j < lda; ++j) {
      double AmA =  A[j+i*lda];
      //printf("%f ", AmA);
    }
    //printf("\n");
  }

  int B1_Xlim = B1_X;
  int B1_Ylim = B1_Y;
  int B1_IDlim = B1_ID;
  int ct = 0;
  /* create first level of blocks */
  for (int i = 0; i < lda; i += B1_X) {
    //int B1_Xlim = (i + B1_X - lda > 0 ? lda % B1_X : B1_X);
    ct += 1;
    for (int j = 0; j < lda; j += B1_Y) {
      //int B1_Ylim = (j + B1_Y - lda > 0 ? lda % B1_Y : B1_Y);
      for (int k = 0; k < lda; k +=B1_ID) { 
	//int B1_IDlim = (k + B1_ID - lda > 0 ? lda % B1_ID : B1_ID);
	for (int kk = 0; kk < B1_IDlim; ++kk) {
	  for (int ii = 0; ii < B1_Xlim; ++ii) {
	    Ablock1[kk+ii*B1_ID] = A[(i+ii)+(k+kk)*lda]; // turn A into row-major
	  }
	  for (int jj = 0; jj < B1_Ylim; ++jj) {
	    Bblock1[kk+jj*B1_ID] = B[(k+kk)+(j+jj)*lda];
	  }
	}
	//printf("matrix Ablock1\n");
	for (int iii = 0; iii < lda; ++iii) {
	  for (int jjj = 0; jjj < lda; ++jjj) {
	    double AmA =  Ablock1[jjj+iii*lda];
	    //printf("%f ", AmA);
	  }
	  //printf("\n");
	}
	make_block1(Ablock1, Bblock1, Cblock1);
	for (int ii = 0; ii < B1_Xlim; ++ii) {
	  for (int jj = 0; jj < B1_Ylim; ++jj) {
	    C[(i+ii)+(j+jj)*lda] += Cblock1[ii+jj*B1_X];
	  }
	}
      }
    }
    //printf("square dgemm = %d\n",ct);
  }
  //printf("lda = %d\n",lda);
  //printf("B1_X = %d\n",B1_X);

}
