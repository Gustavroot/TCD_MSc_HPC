//=============================================================================================
// Name        		: thread3dStl.cu
// Author      		: Jose Refojo
// Version     		:
// Creation date	:	26-02-2014
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will initialize a number of arrays stored in stl vectors,
//			  then it will grab data from each thread (such as thread position inside the block and block),
//			  save it, send it back into the main memory, and print it
//=============================================================================================


#include <iostream>
#include <stdio.h>
#include <vector>

using namespace std;

#define BLOCK_SIZE 2

__global__ void scanTheadInformationGPU(float *threadXIdsGPU,int N,int M,int L) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	int idz=blockIdx.z*blockDim.z+threadIdx.z;

	if ( idx < N ) {
	    if ( idy < M ) {
	    	if ( idz < L ) {
			threadXIdsGPU[idx+idy*N+idz*N*M]=-(idx+idy*N+idz*N*M);
			//threadXIdsGPU[idz+idy*L+idx*M*L]=-(idz+idy*L+idx*M*L);
		}
	    }
	}
}


int main() {
	// pointers to host memory matrices
	std::vector< float >	vector1d;
	std::vector< std::vector< float* > > vector3d;
	float *vector1d_gpu;

	// pointers to device memory matrices
	//float *vectorGPU;
	// N,M and L are the sizes on each dimension
	int N=2,M=3,L=4,totalSize;
	unsigned int ui,uj,uk;

	totalSize=N*M*L;

	// Allocate arrays threadIds and blockIds on host
	vector1d.resize(totalSize);
	vector3d.resize(N, std::vector< float* > (M));

	for (ui=0;ui<N;ui++) {
		for (uj=0;uj<M;uj++) {
			vector3d[ui][uj]=&(vector1d[uj*L+ui*M*L]);
		}
	}

	for (ui=0;ui<N;ui++) {
		for (uj=0;uj<M;uj++) {
			for (uk=0;uk<L;uk++) {
				//vector1d[ui+uj*N+uk*N*M]=ui+uj*N+uk*N*M;
				vector1d[uk+uj*L+ui*M*L]=uk+uj*L+ui*M*L;
			}
		}
	}

	// Allocate arrays threadIdsGPU and blockIdsGPU on device
	cudaMalloc ((void **) &vector1d_gpu, sizeof(float)*N*M*L);

	// Copy data from host memory to device memory
	cudaMemcpy(vector1d_gpu, &(vector1d[0]), sizeof(float)*N*M*L, cudaMemcpyHostToDevice);

	int block_size=BLOCK_SIZE;
	dim3 dimBlock(block_size,block_size,block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1),(M/dimBlock.y) + (!(M%dimBlock.y)?0:1),(L/dimBlock.y) + (!(L%dimBlock.y)?0:1));
	// Call the kernel
	scanTheadInformationGPU <<<dimGrid,dimBlock>>> (vector1d_gpu,N,M,L);


	// Copy data from device memory to host memory
	cudaMemcpy(&(vector1d[0]), vector1d_gpu, sizeof(float)*N*M*L, cudaMemcpyDeviceToHost);

	// Print all the data about the threads
	cout << "vector1d_host: (" << N << "," << M << "," << L << ")" << endl;
	for (ui=0;ui<N;ui++) {
		cout << "vector1d_host, slice in Z " << ui << ":" << endl;
		for (uj=0;uj<M;uj++) {
			for (uk=0;uk<L;uk++) {
				cout << vector1d[uk+uj*L+ui*M*L];
				if (uk==L-1) {	cout << endl;} else { cout << "\t"; }
			}
		}
	}

	cout << "vector3d_host: (" << N << "," << M << "," << L << ")" << endl;
	for (ui=0;ui<N;ui++) {
		cout << "vector3d_host, slice in Z " << ui << ":" << endl;
		for (uj=0;uj<M;uj++) {
			for (uk=0;uk<L;uk++) {
				cout << vector3d[ui][uj][uk];
				if (uk==L-1) {	cout << endl;} else { cout << "\t"; }
			}
		}
	}

/*
		cout << "vector1d_host:" <<endl;
		for (ui=0; ui<numRows; ui++) {
			for (uj=0; uj<numColumns; uj++) {
				cout << input2d_host[ui][uj];
				if (uj==numColumns-1) {	cout << endl;} else { cout << "\t"; }
			}
		}
		cout << endl;
*/
}
