//=============================================================================================
// Name        		: jacobi.cu
// Author      		: Jose Refojo
// Version     		:
// Creation date	:	15-09-10
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will provide an estimate of a function integral in a given interval,
//			  the interval being provided by the user, but the function being fixed.
//=============================================================================================

#define BLOCK_SIZE 8
#define MATRIX_SIZE 100


#include "stdio.h"
#include "time.h"

__global__ void iterateGPUShared (int N,float *A1dGPU,float *bGPU,float *xOldGPU,float *xNewGPU) {	
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int j;
	float sumatory;
	
	if (idx<N) {
		// Does this make any sense? We only need to call A[i][j] once per iteration, anyways...
        	__shared__ float sharedMatrixRow[BLOCK_SIZE][MATRIX_SIZE];
	
		for (j=0;j<N;j++) {
			sharedMatrixRow[threadIdx.x][j] = A1dGPU[j+idx*N];
		}

        	__syncthreads();

		sumatory=bGPU[idx];
		for (j=0;j<N;j++) {
		   if (idx!=j) {
			sumatory-=(sharedMatrixRow[threadIdx.x][j]*xOldGPU[j]);
		   }
		}
		sumatory*= (1.0f/sharedMatrixRow[threadIdx.x][idx]);
		xNewGPU[idx]=sumatory;
	}

}


__global__ void iterateGPU (int N,float *A1dGPU,float *bGPU,float *xOldGPU,float *xNewGPU) {	
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int j;
	float sumatory;

	if (idx<N) {
		sumatory=bGPU[idx];
		for (j=0;j<N;j++) {
		   if (idx!=j) {
			sumatory-=(A1dGPU[j+idx*N]*xOldGPU[j]);
		   }
		}
		sumatory*=(1.0f/A1dGPU[idx+idx*N]);
		xNewGPU[idx]=sumatory;
	}
}
void iterateCPU (int N,float **A,float *b,float *xOld,float *xNew) {
	int i,j;
	float sumatory;
	for (i=0;i<N;i++) {
		sumatory=b[i];
		for (j=0;j<N;j++) {
		   if (i!=j)
			sumatory-=(A[i][j]*xOld[j]);
		}
		sumatory*=(1.0f/A[i][i]);
		xNew[i]=sumatory;
	}
}
bool checkSolution (int N,float **A,float *b,float *xNew) {
	// Calculate r=Ax-b and see how far from [0,,0] it is
	float *r;
	float normMax=-1.E10;
	r = (float*) malloc( N*sizeof(float) );
	int i,j;
	float tmpNorm=0.0f;

	for (i=0;i<N;i++) {
		r[i]=-b[i];
		for (j=0;j<N;j++) {
			r[i]+=A[i][j]*xNew[j];
		}
		tmpNorm += r[i]*r[i];
	}
	free(r);

	printf("checkSolution, tmpNorm: %f\n",tmpNorm);
	if (tmpNorm<1.E-8) {
		return true;
	} else {
		return false;
	}
}
int main() {
	int i,j;
	// Serial Test first:
	int N = MATRIX_SIZE;
	int maxNumberOfIterations=40;

	// Matrix A
	float *A1d;
	float *A1dGPU;
	float **A;
	A1d = (float*) malloc( N*N*sizeof(float) );
	A = (float**) malloc(N*sizeof(float*));
	for (i=0;i<N;i++) {
		A[i]=(&(A1d[i*N]));
	}
	for (i=0;i<N;i++) {
		for (j=0;j<N;j++) {
			if (i!=j) {
				A[i][j] = 0.1;
			} else {
				A[i][j] = 20*N;
			}
		}
	}

	cudaMalloc ((void **) &A1dGPU, sizeof(float)*(N*N));
	cudaMemcpy(A1dGPU, A1d, sizeof(float)*(N*N), cudaMemcpyHostToDevice);

	// Vectors b,xOld,xNew
	float *b,*xOld,*xNew;
	float *bGPU,*xOldGPU,*xNewGPU;
	b = (float*) malloc( N*sizeof(float) );
	for (i=0;i<N;i++) {
		b[i]=i;
	}
	xOld = (float*) malloc( N*sizeof(float) );
	xNew = (float*) malloc( N*sizeof(float) );
	cudaMalloc ((void **) &bGPU, sizeof(float)*N);
	cudaMalloc ((void **) &xOldGPU, sizeof(float)*N);
	cudaMalloc ((void **) &xNewGPU, sizeof(float)*N);

	// We set up the first step of the method as (1,1,...,1)
	for (int i=0;i<N;i++)
		xOld[i]=1;
	
	cudaMemcpy(bGPU   ,    b, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(xNewGPU, xNew, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(xOldGPU, xOld, sizeof(float)*N, cudaMemcpyHostToDevice);

	printf("***********************************************************************************************\n");
	printf("******** This program will provide an estimate of the solution of a linear system problem, ****\n");
	printf("******** using the Jacobi method                                                           ****\n");
	printf("***********************************************************************************************\n");

	clock_t jacobiCPUStart = clock();
	float CPUsolution[2];

	// Iterate
	for (i=0;i<maxNumberOfIterations;i++) {
		printf("======>Iteration %d in CPU\n",i);
		iterateCPU(N,A,b,xOld,xNew);
		if ( checkSolution (N,A,b,xNew) ) {
			// Convergence
			printf("Convergence in %d iterations with the SERIAL code\n",i);
			printf("The solution found was:\n");
			for (j=0;j<N;j++) {
				printf("xNew[%d]=%f\n",j,xNew[j]);
			}
			break;
		} else {
			// No convergence yet, we move xNew to xOld and start again
			for (int i=0;i<N;i++)
				xOld[i]=xNew[i];
			printf("No convergence CPU\n");
		}
	}

	for (i=0;i<maxNumberOfIterations;i++) {
		int block_size=BLOCK_SIZE;
		dim3 dimBlock(block_size);
		dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );
		iterateGPU<<<dimGrid,dimBlock>>>(N,A1dGPU,bGPU,xOldGPU,xNewGPU);
//		iterateGPUShared<<<dimGrid,dimBlock>>>(N,A1dGPU,bGPU,xOldGPU,xNewGPU);

		cudaMemcpy(xNew, xNewGPU, sizeof(float)*N, cudaMemcpyDeviceToHost);

		printf("======>Iteration %d in GPU\n",i);
		if ( checkSolution (N,A,b,xNew) ) {
			// Convergence
			printf("Convergence in %d iterations with the CUDA code\n",i);
			printf("The solution found was:\n");
			for (j=0;j<N;j++) {
				printf("xNew[%d]=%f\n",j,xNew[j]);
			}
			break;
		} else {
			// No convergence yet, we move xNew to xOld and start again
			for (int i=0;i<N;i++)
				xOld[i]=xNew[i];
			cudaMemcpy(xOldGPU, xOld, sizeof(float)*N, cudaMemcpyHostToDevice);
			printf("No convergence GPU\n");
		}
	}
	printf("\n");


	free(A);
	free(A1d);

	free(b);
	free(xOld);
	free(xNew);


}

