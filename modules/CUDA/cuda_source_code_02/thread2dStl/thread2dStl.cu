//=============================================================================================
// Name        		: thread2dStl.cu
// Author      		: Jose Refojo
// Version     		:
// Creation date	:	02-01-11
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will initialize a number of arrays stored in stl vectors,
//			  then it will grab data from each thread (such as thread position inside the block and block),
//			  save it, send it back into the main memory, and print it
//=============================================================================================


#include "stdio.h"
#include <vector>

using namespace std;

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
	std::vector< int* >	threadXIds,threadYIds;
	std::vector< int >	threadXIds1d,threadYIds1d;
	std::vector< int* >	blockXIds,blockYIds;
	std::vector< int >	blockXIds1d;
	std::vector< int >	blockYIds1d;

	// pointers to device memory matrices
	int *threadXIdsGPU, *threadYIdsGPU;
	int *blockXIdsGPU, *blockYIdsGPU;
	// N and M are the total size that we want, N is number of rows and M is number of columns
	int N=3,M=3;
	int i,j;

	// Allocate arrays threadIds and blockIds on host
	// threadIds
	// threadXIds is the pointer to all the array malloced in one dimension
	threadXIds1d.resize(N*M);
	threadYIds1d.resize(N*M);
	// thread*Ids will be just pointers to the one dimension array
	threadXIds.resize(N);
	threadYIds.resize(N);
	for (i=0;i<N;i++) {
		threadXIds[i]=(&(threadXIds1d[i*M]));
		threadYIds[i]=(&(threadYIds1d[i*M]));
	}
	// blockIds
	// blockIds is the pointer to all the array malloced in one dimension
	blockXIds1d.resize(N*M);
	blockYIds1d.resize(N*M);
	// block*Ids will be just pointers to the one dimension array
	blockXIds.resize(N);
	blockYIds.resize(N);
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
	cudaMemcpy(threadXIdsGPU, &(threadXIds1d[0]), sizeof(int)*N*M, cudaMemcpyHostToDevice);
	cudaMemcpy(threadYIdsGPU, &(threadYIds1d[0]), sizeof(int)*N*M, cudaMemcpyHostToDevice);
	cudaMemcpy(blockXIdsGPU,  &(blockXIds1d[0]), sizeof(int)*N*M, cudaMemcpyHostToDevice);
	cudaMemcpy(blockYIdsGPU,  &(blockYIds1d[0]), sizeof(int)*N*M, cudaMemcpyHostToDevice);
*/

	// Compute the execution configuration
	int block_size=2;

	dim3 dimBlock(block_size,block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1),(M/dimBlock.y) + (!(M%dimBlock.y)?0:1) );

	// Scan information from the threads
	scanTheadInformationGPU<<<dimGrid,dimBlock>>>(threadXIdsGPU,threadYIdsGPU,blockXIdsGPU,blockYIdsGPU,N,M);

	// Copy data from device memory to host memory
	cudaMemcpy(&(threadXIds1d[0]), threadXIdsGPU, sizeof(int)*N*M, cudaMemcpyDeviceToHost);
	cudaMemcpy(&(threadYIds1d[0]), threadYIdsGPU, sizeof(int)*N*M, cudaMemcpyDeviceToHost);
	cudaMemcpy(&( blockXIds1d[0]), blockXIdsGPU, sizeof(int)*N*M, cudaMemcpyDeviceToHost);
	cudaMemcpy(&( blockYIds1d[0]), blockYIdsGPU, sizeof(int)*N*M, cudaMemcpyDeviceToHost);

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
	threadXIds.clear();
	threadXIds1d.clear();
	threadYIds.clear();
	threadYIds1d.clear();

	blockXIds.clear();
	blockXIds1d.clear();
	blockYIds.clear();
	blockYIds1d.clear();

	cudaFree(threadXIdsGPU);
	cudaFree(threadYIdsGPU);
	cudaFree(blockXIdsGPU);
	cudaFree(blockYIdsGPU);
}
