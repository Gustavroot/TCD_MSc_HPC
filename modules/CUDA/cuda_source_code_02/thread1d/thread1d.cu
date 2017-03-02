//=============================================================================================
// Name        		: thread1d.cu
// Author      		: Jose Refojo
// Version     		:	26-06-2012
// Creation date	:	18-06-2010
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will initialize a number of arrays,
//			  then it will grab data from each thread (such as thread position inside the block and block),
//			  save it, send it back into the main memory, and print it
//=============================================================================================


#include "stdio.h"

__global__ void scanTheadInformationGPU(int *threadIdsGPU, int *blockIdsGPU,int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx <Ntot ) {
		threadIdsGPU[idx]=threadIdx.x;
		blockIdsGPU[idx]=blockIdx.x;
	}
}


int main() {
	// pointers to host memory
	int *threadIds, *blockIds;
	// pointers to device memory
	int *threadIdsGPU, *blockIdsGPU;
	// N is the total size that we want
	int N=18;
	int i;

	// Allocate arrays threadIds and blockIds on host
	threadIds = (int*) malloc(N*sizeof(int));
	blockIds = (int*) malloc(N*sizeof(int));

	// Allocate arrays threadIdsGPU and blockIdsGPU on device
	cudaMalloc ((void **) &threadIdsGPU, sizeof(int)*N);
	cudaMalloc ((void **) &blockIdsGPU, sizeof(int)*N);
/*
	// Copy data from host memory to device memory (not needed, but this is how you do it)
	cudaMemcpy(threadIdsGPU, threadIds, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(blockIdsGPU, blockIds, sizeof(int)*N, cudaMemcpyHostToDevice);
*/

	// Compute the execution configuration
	int block_size=8;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// Scan information from the threads
	scanTheadInformationGPU<<<dimGrid,dimBlock>>>(threadIdsGPU, blockIdsGPU, N);

	// Copy data from device memory to host memory
	cudaMemcpy(threadIds, threadIdsGPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(blockIds, blockIdsGPU, sizeof(int)*N, cudaMemcpyDeviceToHost);

	// Print all the data about the threads
	printf(" dimGrid=%d\n",dimGrid.x);
	for (i=0; i<N; i++) {
	       printf(" threadIds[%d]=%d\n",i,threadIds[i]);
	}
	for (i=0; i<N; i++) {
	       printf(" blockIds[%d]=%d\n",i,blockIds[i]);
	}

	// Free the memory
	free(threadIds);
	free(blockIds); 

	cudaFree(threadIdsGPU);
	cudaFree(blockIdsGPU);
}
