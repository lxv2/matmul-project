const char* dgemm_desc = "My awesome dgemm.";

/* for the L2 cache */
#define B1_X  ((const int) 16)
#define B1_ID ((const int) 16) /* inner dimension */
#define B1_Y  ((const int) 16)

/* for the L1 cache */
#define B2_X  ((const int) 8)
#define B2_ID ((const int) 8) /* inner dimension */
#define B2_Y  ((const int) 8)

/* for the registers */
#define B3_X  ((const int) 2)
#define B3_ID ((const int) 2) /* inner dimension */
#define B3_Y  ((const int) 2)

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
      double cij = Cblock3[i+j*B3_X];
#pragma simd
      for (k = 0; k < B3_ID; ++k) {
	cij += Ablock3[k+i*B3_ID] * Bblock3[k+j*B3_ID];
      }
      Cblock3[i+j*B3_X] = cij;
    }
  }
}
    
inline void make_block2(const double Ablock2[B2_X*B2_ID], const double Bblock2[B2_Y*B2_ID],
			double Cblock2[B2_X*B2_Y], double Ablock3[B3_X*B3_ID], double Bblock3[B3_Y*B3_ID],
			double Cblock3[B3_X*B3_Y])
{
  /* send to basic dgemm solver */
  for (int i = 0; i < B2_X ; i += B3_X) {
    for (int j = 0; j < B2_Y ; j += B3_Y) {
      Cblock3[B3_X*B3_Y] = 0;
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
	basic_dgemm(Ablock3, Bblock3, Cblock3);
	// copy outputs from block series 3 and reset Cblock3
	for (int ii = 0; ii < B3_X; ++ii) {
	  for (int jj = 0; jj < B3_Y; ++jj) {
	    Cblock2[(i+ii)+(j+jj)*B2_X] += Cblock3[ii+jj*B3_X];
	    Cblock3[ii+jj*B3_X] = 0;
	  }
	}
      }
    }
  }
}

inline void make_block1(const double Ablock1[B1_X*B1_ID], const double Bblock1[B1_Y*B1_ID],
			double Cblock1[B1_X*B1_Y], double Ablock2[B2_X*B2_ID], double Bblock2[B2_Y*B2_ID],
			double Cblock2[B2_X*B2_Y], double Ablock3[B3_X*B3_ID], double Bblock3[B3_Y*B3_ID],
			double Cblock3[B3_X*B3_Y])
{
  for (int i = 0; i < B1_X ; i +=B2_X) {
    for (int j = 0; j < B1_Y ; j +=B2_Y) {
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
	make_block2(Ablock2, Bblock2, Cblock2, Ablock3, Bblock3, Cblock3);
	// copy outputs from block series 2
	for (int ii = 0; ii < B2_X; ++ii) {
	  for (int jj = 0; jj < B2_Y; ++jj) {
	    Cblock1[(i+ii)+(j+jj)*B1_X] += Cblock2[ii+jj*B2_X];
	    Cblock2[ii+jj*B2_X] = 0;
	  }
	}
      }
    }
  }
}
    
void square_dgemm(const int lda, const double * restrict A, const double * restrict B, double * restrict C)
{

  /* create blocks for each level */
  double Ablock1[B1_X*B1_ID] = {0};
  double Bblock1[B1_Y*B1_ID] = {0};
  double Cblock1[B1_X*B1_Y ] = {0};

  /* create second level of blocks */
  double Ablock2[B2_X*B2_ID] = {0};
  double Bblock2[B2_Y*B2_ID] = {0};
  double Cblock2[B2_X*B2_Y ] = {0};

  /* create third level of blocks */
  double Ablock3[B3_X*B3_ID] = {0};
  double Bblock3[B3_Y*B3_ID] = {0};
  double Cblock3[B3_X*B3_Y ] = {0};

  /* create first level of blocks, do blocks with no remainder */
  for (int i = 0; i < lda; i += B1_X) {
    int B1_Xlim = (i + B1_X - lda > 0 ? lda % B1_X : B1_X);
    for (int j = 0; j < lda; j += B1_Y) {
      int B1_Ylim = (j + B1_Y - lda > 0 ? lda % B1_Y : B1_Y);
      for (int k = 0; k < lda; k +=B1_ID) { 
	int B1_IDlim = ( ((k + B1_ID - lda) > 0) ? lda % B1_ID : B1_ID);
	for (int kk = 0; kk < B1_IDlim; ++kk) {
	  for (int ii = 0; ii < B1_Xlim; ++ii) {
	    Ablock1[kk+ii*B1_ID] = A[(i+ii)+(k+kk)*lda]; // turn A into row-major
	  }
	  for (int ii = B1_Xlim; ii < B1_X; ++ii) {
	    Ablock1[kk+ii*B1_ID] = 0;
	  }
	  for (int jj = 0; jj < B1_Ylim; ++jj) {
	    Bblock1[kk+jj*B1_ID] = B[(k+kk)+(j+jj)*lda];
	  }
	  for (int jj = B1_Ylim; jj < B1_Y; ++jj) {
	    Bblock1[kk+jj*B1_ID] = 0;
	  }
	}
	for (int kk = B1_IDlim; kk < B1_ID; ++kk) {
	  for (int ii = 0; ii < B1_X; ++ii) {
	    Ablock1[kk+ii*B1_ID] = 0;
	  }
	  for (int jj = 0; jj < B1_Y; ++jj) {
	    Bblock1[kk+jj*B1_ID] = 0;
	  }
	}
	make_block1(Ablock1, Bblock1, Cblock1, Ablock2, Bblock2, Cblock2, Ablock3, Bblock3, Cblock3);
	for (int ii = 0; ii < B1_Xlim; ++ii) {
	  for (int jj = 0; jj < B1_Ylim; ++jj) {
	    C[(i+ii)+(j+jj)*lda] += Cblock1[ii+jj*B1_X];
	    Cblock1[ii+jj*B1_X] = 0;
	  }
	}
      }
    }
  }
}
