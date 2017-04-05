#include "stdio.h"
#include "sys_evol_gpu.h"
#include "math.h"


//to keep float accuracy, execute with
//values n*m ~ 10^6 or less

//'SPLIT' is the actual number of threads
//#define SPLIT 1024


typedef float VAR_TYPE;



__global__ void system_evolution_gpu(VAR_TYPE* M_orig, VAR_TYPE* M_next_orig, VAR_TYPE* thermostat_buff, int n, int m, int nr_threads, int nr_iterations){

  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if(idx >= nr_threads){return;}

  //threadIdx.x gives the position of the thread within the block

  int j, k, init_j;
  //buff pointer to swap matrices
  VAR_TYPE *M_buff, *M, *M_next;
  //in case of use of global memory, shift data
  int shift = 0;

  //in case there is enough shared memory
  if(m<=2048){
    //dynamically allocating shared memory, and move one
    //row (in this case equal to one block) to that new block
    extern __shared__ VAR_TYPE M_total[];
    //'sm' stands for Shared Memory
    //split of allocated shared memory
    M = M_total;
    M_next = M_total + m;
  }
  //but if not enough shared memory, use global
  else{
    M = M_orig;
    M_next = M_next_orig;
    shift = blockIdx.x*m;
  }

  if(m<=2048){
    //copying from global to shared memory
    for(j=threadIdx.x*(m/blockDim.x); j<(threadIdx.x+1)*(m/blockDim.x); j++){
      M[j] = M_orig[blockIdx.x*m + j];
      M_next[j] = M[j];
    }
  }

  //TODO: analytically evaluate the need for next line
  __syncthreads();

  //i is now blockIdx.x

  if(threadIdx.x == 0){
    init_j = 2;
  }
  else if(threadIdx.x == 1 && m%blockDim.x == 0){
    init_j = 2;
  }
  else{
    init_j = threadIdx.x*(m/blockDim.x);
  }

  for(k=0; k<nr_iterations; k++){

    //omit modifications over j = {0, 1}
    for(j=init_j; j<(threadIdx.x+1)*(m/blockDim.x); j++){
      //when j equals one in {m-1, m-2}, then apply boundary conditions
      if(j == m-2){
        M_next[shift + j] = (1/( (VAR_TYPE)(5.0) )) * (1.9*M[shift + (j-2)] +
                1.5*M[shift + (j-1)] + M[shift + j] + 0.5*M[shift + (j+1)]
                + 0.1*M[shift + 0] );
                //+ 0.1*M[blockIdx.x*m + 0] );
      }
      else if(j == m-1){
        M_next[shift + j] = (1/( (VAR_TYPE)(5.0) )) * (1.9*M[shift + (j-2)] +
                1.5*M[shift + (j-1)] + M[shift + j] + 0.5*M[shift + 0]
                + 0.1*M[shift + 1] );
      }
      else{
        M_next[shift + j] = (1/( (VAR_TYPE)(5.0) )) * (1.9*M[shift + (j-2)] +
                1.5*M[shift + (j-1)] + M[shift + j] + 0.5*M[shift + (j+1)]
                + 0.1*M[shift + (j+2)] );
      }
    }

    //Sync within each block needed, as there is
    //no vertical heat disipation
    __syncthreads();

    //swap matrices
    M_buff = M;
    M = M_next;
    M_next = M_buff;
  }

  if(m<=2048){
    //copy data back to global memory
    for(j=threadIdx.x*(m/blockDim.x); j<(threadIdx.x+1)*(m/blockDim.x); j++){
      M_orig[blockIdx.x*m + j] = M[j];
    }
  }

  //the 1st thread of each block processes/averages T
  if(idx%blockDim.x == 0){
    thermostat_buff[blockIdx.x] = 0;
    for(j=0; j<m; j++){
      //atomicAdd(thermostat_buff + blockIdx.x, M[shift + j]);
      thermostat_buff[blockIdx.x] += M[shift + j];
    }
  }
}





//'t_p_b' = 'threads per block'
extern VAR_TYPE cyl_rad_cu(VAR_TYPE* M, VAR_TYPE* M_next, VAR_TYPE* thermostat, int n, int m, double* d_t, int to_time, int nr_iterations, int t_p_b){

  struct timeval begin, end;
  double d_t_buff;

  //general purpose counter
  int i;

  //total number of elements in matrix
  int N = n*m;

  //pointer to matrix at gpu
  VAR_TYPE *M_d, *M_next_d;
  //thermostat at gpu
  VAR_TYPE* thermostat_d;

  //total number of threads.. one block per row
  //-- the following re-factor accounts for shared memory limitations --
  int nr_threads = n*t_p_b;

  VAR_TYPE* thermostat_buff = (VAR_TYPE*)malloc(n*sizeof(VAR_TYPE));

  //copy: from host to device
  gettimeofday(&begin, NULL);

  cudaMalloc( (void**) &M_d, sizeof(VAR_TYPE)*N );
  cudaMalloc( (void**) &M_next_d, sizeof(VAR_TYPE)*N );
  //each row is a block
  cudaMalloc( (void**) &thermostat_d, sizeof(VAR_TYPE)*n );

  gettimeofday(&end, NULL);

  if(to_time){
    *d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec -
                begin.tv_usec)/1000000.0);
    printf("\ncudaMalloc time: %f\n", *d_t);
  }

  //copy: from host to device
  gettimeofday(&begin, NULL);

  cudaMemcpy(M_d, M, sizeof(VAR_TYPE)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(M_next_d, M_next, sizeof(VAR_TYPE)*N, cudaMemcpyHostToDevice);

  gettimeofday(&end, NULL);

  if(to_time){
    *d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec -
		begin.tv_usec)/1000000.0);
    printf("\ncudaMemcpy time: %f\n", *d_t);
  }

  int block_size;
  //configuration for execution at gpu
  gettimeofday(&begin, NULL);

  //re_factor is the block size in the 'y' direction
  block_size = t_p_b;
  dim3 dimBlock(block_size);
  //dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );
  dim3 dimGrid(nr_threads/dimBlock.x);

  printf("\nGPU config:\n");
  printf("\t** block size: %d\n", block_size);
  printf("\t** number of threads: %d\n", nr_threads);
  printf("\t** number of blocks: %d\n\n", nr_threads/block_size);

  gettimeofday(&end, NULL);

  if(to_time){
    *d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec -
		begin.tv_usec)/1000000.0);
    printf("gpu config time: %f\n", *d_t);
  }

  //re-factor due to shared memory issues
  int re_factor = 1;
  if(m>2048){
    if(m%2048 == 0){
      re_factor *= m/2048;
    }
    else{
      re_factor *= (int)(((float)(m))/((float)(2048.0))) + 1;
    }
  }

  //call to functions executing on gpu
  gettimeofday(&begin, NULL);

  //BEGINNING of timing with CUDA events

  //create events
  cudaEvent_t start, finish;
  cudaEventCreate(&start);
  cudaEventCreate(&finish);

  //record events around kernel launch
  cudaEventRecord(start, 0);
  system_evolution_gpu<<<dimGrid, dimBlock, (2*m/re_factor)*sizeof(VAR_TYPE)>>>(M_d, M_next_d, thermostat_d, n, m, nr_threads, nr_iterations);
  cudaEventRecord(finish, 0);

  //synchronize
  cudaEventSynchronize(start);
  cudaEventSynchronize(finish);

  //calculate time
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, finish);

  if(to_time){
    printf("\nEXEC TIME, parallel CUDA-events (microseconds): %f\n\n", elapsedTime);
  }

  //END of timing with CUDA events

  gettimeofday(&end, NULL);

  if(to_time){
  *d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec -
		begin.tv_usec)/1000000.0);
    printf("evolution of system processed! Exec time: %f\n", *d_t);
  }

  d_t_buff = *d_t;

  //copy back from device to host
  gettimeofday(&begin, NULL);
  cudaMemcpy(M, M_d, sizeof(VAR_TYPE)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(thermostat_buff, thermostat_d, sizeof(VAR_TYPE)*n, cudaMemcpyDeviceToHost);

  gettimeofday(&end, NULL);
  if(to_time){
    *d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec -
		begin.tv_usec)/1000000.0);
    printf("copy-back-to-host time: %f\n", *d_t);
  }

  *d_t = d_t_buff;

  //process 'thermostat_buff', to return averages appropriately into 'thermostat'
  for(i=0; i<n; i++){
    thermostat[i] = thermostat_buff[i];
  }

  cudaFree(M_d);
  cudaFree(thermostat_d);
  cudaFree(M_next_d);
  free(thermostat_buff);
}
