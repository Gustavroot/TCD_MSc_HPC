//=============================================================================================
// Name        		: thread2d.cu
// Author      		: Jose Refojo
// Version     		:	29-06-2012
// Creation date	:	18-06-2010
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will initialize a number of arrays,
//			  then it will grab data from each thread (such as thread position inside the block and block),
//			  save it, send it back into the main memory, and print it
//=============================================================================================


#include "stdio.h"

__global__ void scanTheadInformationGPU(int *threadXIdsGPU,int *threadYIdsGPU,int *blockXIdsGPU,int *blockYIdsGPU,int N,int M) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;

	if ( idx < N ) {
	    if ( idy < M ) {
		threadXIdsGPU[idx+idy*N]=threadIdx.x;
		threadYIdsGPU[idx+idy*N]=threadIdx.y;
		blockXIdsGPU[idx+idy*N]=blockIdx.x;
		blockYIdsGPU[idx+idy*N]=blockIdx.y;
	    }
	}
}


int main() {
	// pointers to host memory matrices
	int **threadXIds, **threadYIds;
	int *threadXIds1d = NULL;
	int *threadYIds1d = NULL;
	int **blockXIds, **blockYIds;
	int *blockXIds1d = NULL;
	int *blockYIds1d = NULL;

	// pointers to device memory matrices
	int *threadXIdsGPU, *threadYIdsGPU;
	int *blockXIdsGPU, *blockYIdsGPU;
	// N and M are the total size that we want, N is number of rows and M is number of columns
	int N=3,M=3;
	int i,j;

	// Allocate arrays threadIds and blockIds on host
	// threadIds
	// threadXIds is the pointer to all the array malloced in one dimension
	threadXIds1d = (int*) malloc( (N)*(M)*sizeof(int) );
	threadYIds1d = (int*) malloc( (N)*(M)*sizeof(int) );
	// thread*Ids will be just pointers to the one dimension array
	threadXIds = (int**) malloc((N)*sizeof(int*));
	threadYIds = (int**) malloc((N)*sizeof(int*));
	for (i=0;i<N;i++) {
		threadXIds[i]=(&(threadXIds1d[i*M]));
		threadYIds[i]=(&(threadYIds1d[i*M]));
	}
	// blockIds
	// blockIds is the pointer to all the array malloced in one dimension
	blockXIds1d = (int*) malloc( (N)*(M)*sizeof(int) );
	blockYIds1d = (int*) malloc( (N)*(M)*sizeof(int) );
	// block*Ids will be just pointers to the one dimension array
	blockXIds = (int**) malloc((N)*sizeof(int*));
	blockYIds = (int**) malloc((N)*sizeof(int*));
	for (i=0;i<N;i++) {
		blockXIds[i]=(&(blockXIds1d[i*M]));
		blockYIds[i]=(&(blockYIds1d[i*M]));
	}


	// Allocate arrays threadIdsGPU and blockIdsGPU on device
	cudaMalloc ((void **) &threadXIdsGPU, sizeof(int)*N*M);
	cudaMalloc ((void **) &threadYIdsGPU, sizeof(int)*N*M);
	cudaMalloc ((void **) &blockXIdsGPU, sizeof(int)*N*M);
	cudaMalloc ((void **) &blockYIdsGPU, sizeof(int)*N*M);
/*
	// Copy data from host memory to device memory (not needed)
	cudaMemcpy(threadXIdsGPU, threadXIds1d, sizeof(int)*N*M, cudaMemcpyHostToDevice);
	cudaMemcpy(threadYIdsGPU, threadYIds1d, sizeof(int)*N*M, cudaMemcpyHostToDevice);
	cudaMemcpy(blockXIdsGPU, blockXIds1d, sizeof(int)*N*M, cudaMemcpyHostToDevice);
	cudaMemcpy(blockYIdsGPU, blockYIds1d, sizeof(int)*N*M, cudaMemcpyHostToDevice);
*/

	// Compute the execution configuration
	int block_size=2;

	dim3 dimBlock(block_size,block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1),(M/dimBlock.y) + (!(M%dimBlock.y)?0:1) );

	// Scan information from the threads
	scanTheadInformationGPU<<<dimGrid,dimBlock>>>(threadXIdsGPU,threadYIdsGPU,blockXIdsGPU,blockYIdsGPU,N,M);

	// Copy data from device memory to host memory
	cudaMemcpy(threadXIds1d, threadXIdsGPU, sizeof(int)*N*M, cudaMemcpyDeviceToHost);
	cudaMemcpy(threadYIds1d, threadYIdsGPU, sizeof(int)*N*M, cudaMemcpyDeviceToHost);
	cudaMemcpy(blockXIds1d, blockXIdsGPU, sizeof(int)*N*M, cudaMemcpyDeviceToHost);
	cudaMemcpy(blockYIds1d, blockYIdsGPU, sizeof(int)*N*M, cudaMemcpyDeviceToHost);

	// Print all the data about the threads
	printf(" dimGrid = %d %d\n",dimGrid.x,dimGrid.y);
	for (i=0; i<N; i++) {
		for (j=0; j<M; j++) {
			printf(" threadIds[%d][%d]= %d , %d\n",i,j,threadXIds[i][j],threadYIds[i][j]);
		}
	}
	for (i=0; i<N; i++) {
		for (j=0; j<M; j++) {
			printf(" blockIds[%d][%d]= %d , %d\n",i,j,blockXIds[i][j],blockYIds[i][j]);
		}
	}

	// Free the memory
	free(threadXIds);
	free(threadXIds1d);
	free(threadYIds);
	free(threadYIds1d);

	free(blockXIds); 
	free(blockXIds1d); 
	free(blockYIds); 
	free(blockYIds1d); 

	cudaFree(threadXIdsGPU);
	cudaFree(threadYIdsGPU);
	cudaFree(blockXIdsGPU);
	cudaFree(blockYIdsGPU);
}
