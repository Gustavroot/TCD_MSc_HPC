//=============================================================================================
// Name        		: thread3D.cu
// Author      		: Jose Refojo
// Version     		:	29-06-2012
// Creation date	:	18-06-2010
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will initialize a number of arrays,
//			  then it will grab data from each thread (such as thread position inside the block and block),
//			  save it, send it back into the main memory, and print it
//=============================================================================================


#include "stdio.h"

__global__ void scanTheadInformationGPU(int *threadXIdsGPU,int *threadYIdsGPU,int *threadZIdsGPU,int *blockXIdsGPU,int *blockYIdsGPU,int *blockZIdsGPU,int N,int M,int L) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	int idz=threadIdx.z;

	if ( idx < N ) {
	    if ( idy < M ) {
	    	if ( idz < L ) {
			threadXIdsGPU[idx+idy*N+idz*N*M]=threadIdx.x;
			threadYIdsGPU[idx+idy*N+idz*N*M]=threadIdx.y;
			threadZIdsGPU[idx+idy*N+idz*N*M]=threadIdx.z;
			blockXIdsGPU[idx+idy*N+idz*N*M]=blockIdx.x;
			blockYIdsGPU[idx+idy*N+idz*N*M]=blockIdx.y;
			blockZIdsGPU[idx+idy*N+idz*N*M]=blockIdx.z;
		}
	    }
	}
}


int main() {
	// pointers to host memory matrices
	int ***threadXIds, ***threadYIds, ***threadZIds;
	int *threadXIds1d = NULL;
	int *threadYIds1d = NULL;
	int *threadZIds1d = NULL;
	int ***blockXIds, ***blockYIds, ***blockZIds;
	int *blockXIds1d = NULL;
	int *blockYIds1d = NULL;
	int *blockZIds1d = NULL;

	// pointers to device memory matrices
	int *threadXIdsGPU, *threadYIdsGPU, *threadZIdsGPU;
	int *blockXIdsGPU, *blockYIdsGPU, *blockZIdsGPU;
	// N and M are the total size that we want, N is number of rows and M is number of columns
	int N=4,M=4,L=4;
	int i,j,k;

	// Allocate arrays threadIds and blockIds on host
	// threadIds
	// threadXIds is the pointer to all the array malloced in one dimension
	threadXIds1d = (int*) malloc( N*M*L*sizeof(int) );
	threadYIds1d = (int*) malloc( N*M*L*sizeof(int) );
	threadZIds1d = (int*) malloc( N*M*L*sizeof(int) );
	// thread*Ids will be just pointers to the one dimension array
	threadXIds = (int***) malloc(N*sizeof(int**));
	threadYIds = (int***) malloc(N*sizeof(int**));
	threadZIds = (int***) malloc(N*sizeof(int**));
	for (i=0;i<N;i++) {
		int **tmpPointerX = (int**) malloc(M*sizeof(int*));
		int **tmpPointerY = (int**) malloc(M*sizeof(int*));
		int **tmpPointerZ = (int**) malloc(M*sizeof(int*));
		for (j=0;j<M;j++) {
			tmpPointerX[j]=(&(threadXIds1d[i*M*L+j*L]));
			tmpPointerY[j]=(&(threadYIds1d[i*M*L+j*L]));
			tmpPointerZ[j]=(&(threadZIds1d[i*M*L+j*L]));
		}
		threadXIds[i]=tmpPointerX;
		threadYIds[i]=tmpPointerY;
		threadZIds[i]=tmpPointerZ;
	}

	// blockIds
	// blockIds is the pointer to all the array malloced in one dimension
	blockXIds1d = (int*) malloc( N*M*L*sizeof(int) );
	blockYIds1d = (int*) malloc( N*M*L*sizeof(int) );
	blockZIds1d = (int*) malloc( N*M*L*sizeof(int) );
	// block*Ids will be just pointers to the one dimension array
	blockXIds = (int***) malloc(N*sizeof(int**));
	blockYIds = (int***) malloc(N*sizeof(int**));
	blockZIds = (int***) malloc(N*sizeof(int**));
	for (i=0;i<N;i++) {
		int **tmpPointerX = (int**) malloc(M*sizeof(int*));
		int **tmpPointerY = (int**) malloc(M*sizeof(int*));
		int **tmpPointerZ = (int**) malloc(M*sizeof(int*));
		for (j=0;j<M;j++) {
			tmpPointerX[j]=(&(blockXIds1d[i*M*L+j*L]));
			tmpPointerY[j]=(&(blockYIds1d[i*M*L+j*L]));
			tmpPointerZ[j]=(&(blockZIds1d[i*M*L+j*L]));
		}
		blockXIds[i]=tmpPointerX;
		blockYIds[i]=tmpPointerY;
		blockZIds[i]=tmpPointerZ;
	}

	// Allocate arrays threadIdsGPU and blockIdsGPU on device
	cudaMalloc ((void **) &threadXIdsGPU, sizeof(int)*N*M*L);
	cudaMalloc ((void **) &threadYIdsGPU, sizeof(int)*N*M*L);
	cudaMalloc ((void **) &threadZIdsGPU, sizeof(int)*N*M*L);
	cudaMalloc ((void **) &blockXIdsGPU, sizeof(int)*N*M*L);
	cudaMalloc ((void **) &blockYIdsGPU, sizeof(int)*N*M*L);
	cudaMalloc ((void **) &blockZIdsGPU, sizeof(int)*N*M*L);


	// Copy data from host memory to device memory (not needed)
	cudaMemcpy(threadXIdsGPU, threadXIds1d, sizeof(int)*N*M*L, cudaMemcpyHostToDevice);
	cudaMemcpy(threadYIdsGPU, threadYIds1d, sizeof(int)*N*M*L, cudaMemcpyHostToDevice);
	cudaMemcpy(threadZIdsGPU, threadZIds1d, sizeof(int)*N*M*L, cudaMemcpyHostToDevice);
	cudaMemcpy(blockXIdsGPU, blockXIds1d, sizeof(int)*N*M*L, cudaMemcpyHostToDevice);
	cudaMemcpy(blockYIdsGPU, blockYIds1d, sizeof(int)*N*M*L, cudaMemcpyHostToDevice);
	cudaMemcpy(blockZIdsGPU, blockZIds1d, sizeof(int)*N*M*L, cudaMemcpyHostToDevice);



	// Compute the execution configuration
	int block_size=2;
	// Block size has to be L in Z since CUDA does not allow 3d grids
	dim3 dimBlock(block_size,block_size,L);
	// Which is why we have to use "1" as the third dimension here:
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1),(M/dimBlock.y) + (!(M%dimBlock.y)?0:1) ,1);

	// Scan information from the threads
	scanTheadInformationGPU<<<dimGrid,dimBlock>>>(threadXIdsGPU,threadYIdsGPU,threadZIdsGPU,blockXIdsGPU,blockYIdsGPU,blockZIdsGPU,N,M,L);

	// Copy data from device memory to host memory
	cudaMemcpy(threadXIds1d, threadXIdsGPU, sizeof(int)*N*M*L, cudaMemcpyDeviceToHost);
	cudaMemcpy(threadYIds1d, threadYIdsGPU, sizeof(int)*N*M*L, cudaMemcpyDeviceToHost);
	cudaMemcpy(threadZIds1d, threadZIdsGPU, sizeof(int)*N*M*L, cudaMemcpyDeviceToHost);
	cudaMemcpy(blockXIds1d, blockXIdsGPU, sizeof(int)*N*M*L, cudaMemcpyDeviceToHost);
	cudaMemcpy(blockYIds1d, blockYIdsGPU, sizeof(int)*N*M*L, cudaMemcpyDeviceToHost);
	cudaMemcpy(blockZIds1d, blockZIdsGPU, sizeof(int)*N*M*L, cudaMemcpyDeviceToHost);

	// Print all the data about the threads
	printf(" dimGrid = %d %d %d\n",dimGrid.x,dimGrid.y,dimGrid.z);
	for (i=0; i<N; i++) {
		for (j=0; j<M; j++) {
			for (k=0; k<L; k++) {
				printf(" threadIds[%d][%d][%d]= %d , %d, %d\n",i,j,k,threadXIds[i][j][k],threadYIds[i][j][k],threadZIds[i][j][k]);
			}
		}
	}

	for (i=0; i<N; i++) {
		for (j=0; j<M; j++) {
			for (k=0; k<L; k++) {
				printf(" blockIds[%d][%d][%d]= %d , %d, %d\n",i,j,k,blockXIds[i][j][k],blockYIds[i][j][k],blockZIds[i][j][k]);
			}
		}
	}

	// Free the memory
	// Free the 1d
	free(threadXIds1d);
	free(threadYIds1d);
	free(threadZIds1d);

	free(blockXIds1d);
	free(blockYIds1d);
	free(blockZIds1d);
	// Free the 2d
	for (i=0;i<N;i++) {
		free(threadXIds[i]);
		free(threadYIds[i]);
		free(threadZIds[i]);

		free(blockXIds[i]);
		free(blockYIds[i]);
		free(blockZIds[i]);
	}
	// Free the 3d
	free(threadXIds);
	free(threadYIds);
	free(threadZIds);

	free(blockXIds);
	free(blockYIds);
	free(blockZIds);

	cudaFree(threadXIdsGPU);
	cudaFree(threadYIdsGPU);
	cudaFree(threadZIdsGPU);
	cudaFree(blockXIdsGPU);
	cudaFree(blockYIdsGPU);
	cudaFree(blockZIdsGPU);

}
