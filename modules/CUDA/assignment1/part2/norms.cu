#include "stdio.h"
#include "norms.h"
#include "math.h"

//to keep float accuracy, execute with
//values n*m ~ 10^6 or less

//'SPLIT' is the actual number of threads
#define SPLIT 1024

typedef float VAR_TYPE;



__global__ void max_norm_gpu(VAR_TYPE* M, VAR_TYPE* norm, int N, int nr_threads){

  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int i, max;

  VAR_TYPE buff_float; norm[idx] = 0;

  if(idx >= nr_threads){return;}
  if(idx == nr_threads-1){
    max = N;
  }else{
    max = (idx+1)*(N/nr_threads);
  }

  for(i=idx*(N/nr_threads); i<max; i++){
    buff_float = M[i];
    if(buff_float < 0){
      buff_float = -buff_float;
    }
    if(buff_float > norm[idx]){
      norm[idx] = buff_float;
    }
  }
}

__global__ void frobenius_norm_gpu(VAR_TYPE* M, VAR_TYPE* norm, int N, int nr_threads){

  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int i, max;

  norm[idx] = 0;

  if(idx >= nr_threads){return;}
  if(idx == nr_threads-1){
    max = N;
  }else{
    max = (idx+1)*(N/nr_threads);
  }

  for(i=idx*(N/nr_threads); i<max; i++){
    norm[idx] += M[i]*M[i];
  }

  __syncthreads();

}

__global__ void one_norm_gpu(VAR_TYPE* M, VAR_TYPE* norm, int N, int nr_threads, int n_rows, int m_cols){

  //each thread executes one column, and if the number of
  //colums is greater than the number of threads, then rotate threads

  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int i, j;
  VAR_TYPE buff;

  if(idx >= nr_threads){return;}
  for(j=idx; j<m_cols; j+=nr_threads){

    norm[j] = 0;
    
    //now, summation over each column
    for(i=0; i<n_rows; i++){
      buff = M[j+i*m_cols];
      if(buff < 0){
        buff = -buff;
      }
      norm[j] += buff;
    }

  }
}

__global__ void infinite_norm_gpu(VAR_TYPE* M, VAR_TYPE* norm, int N, int nr_threads, int n_rows, int m_cols){

  //each thread executes one column, and if the number of
  //colums is greater than the number of threads, then rotate threads

  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int i, j;
  VAR_TYPE buff;

  if(idx >= nr_threads){return;}
  for(j=idx; j<m_cols; j+=nr_threads){

    norm[j] = 0;

    //now, summation over each column
    for(i=j*n_rows; i<(j+1)*n_rows; i++){
      buff = M[i];
      if(buff < 0){
        buff = -buff;
      }
      norm[j] += buff;
    }

  }
  
}

//'t_p_b' = 'threads per block'
extern VAR_TYPE norms_cu(VAR_TYPE* M, int n, int m, double* d_t, int which, int to_time, int t_p_b){
  struct timeval begin, end;
  double d_t_buff;

  //general purpose counter
  int i;

  //total number of elements in matrix
  int N = n*m;

  //pointer to matrix in memory
  VAR_TYPE* M_d;
  //value of norm at cpu
  VAR_TYPE norm_val = 0;
  //value of norm at gpu
  VAR_TYPE* norm_ptr;
  if(which == 2){
    norm_ptr = (VAR_TYPE*) malloc(sizeof(VAR_TYPE)*m);
  }
  else if(which == 3){
    norm_ptr = (VAR_TYPE*) malloc(sizeof(VAR_TYPE)*n);
  }
  else{
    norm_ptr = (VAR_TYPE*) malloc(sizeof(VAR_TYPE)*SPLIT);
  }
  VAR_TYPE *norm_d;

  cudaMalloc( (void**) &M_d, sizeof(VAR_TYPE)*N );
  cudaMalloc( (void**) &norm_d, sizeof(VAR_TYPE)*SPLIT );

  //copy: from host to device
  gettimeofday(&begin, NULL);
  cudaMemcpy(M_d, M, sizeof(VAR_TYPE)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(norm_d, norm_ptr, sizeof(VAR_TYPE)*SPLIT, cudaMemcpyHostToDevice);
  gettimeofday(&end, NULL);
  if(to_time){
    *d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec -
		begin.tv_usec)/1000000.0);
    printf("\ncudaMemcpy time: %f\n", *d_t);
  }

  int block_size;
  //configuration for execution at gpu
  gettimeofday(&begin, NULL);

  block_size = t_p_b;
  dim3 dimBlock(block_size);
  //dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );
  dim3 dimGrid( SPLIT / block_size );

  gettimeofday(&end, NULL);
  if(to_time){
    *d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec -
		begin.tv_usec)/1000000.0);
    printf("gpu config time: %f\n", *d_t);
  }

  //call to functions executing on gpu
  gettimeofday(&begin, NULL);
  if(which == 0){
    max_norm_gpu<<<dimGrid,dimBlock>>>(M_d, norm_d, N, SPLIT);
  }
  else if(which == 1){
    frobenius_norm_gpu<<<dimGrid,dimBlock>>>(M_d, norm_d, N, SPLIT);
  }
  else if(which == 2){
    one_norm_gpu<<<dimGrid,dimBlock>>>(M_d, norm_d, N, SPLIT, n, m);
  }
  else{
    infinite_norm_gpu<<<dimGrid,dimBlock>>>(M_d, norm_d, N, SPLIT, n, m);
  }
  gettimeofday(&end, NULL);
  if(to_time){
  *d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec -
		begin.tv_usec)/1000000.0);
    printf("norm processed!\n");
  }

  d_t_buff = *d_t;

  //copy back from device to host
  gettimeofday(&begin, NULL);
  cudaMemcpy(norm_ptr, norm_d, sizeof(VAR_TYPE)*SPLIT, cudaMemcpyDeviceToHost);
  gettimeofday(&end, NULL);
  if(to_time){
    *d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec -
		begin.tv_usec)/1000000.0);
    printf("copy-back-to-host time: %f\n", *d_t);
  }

  *d_t = d_t_buff;

  norm_val = 0;
  //max norm
  if(which == 0){
    //find max float within array 'norm_ptr' and set as value for 'norm_val'
    for(i=0; i<SPLIT; i++){
      if(norm_ptr[i] > norm_val){
        norm_val = norm_ptr[i];
      }
    }
  }
  //frobenius norm
  else if(which == 1){
    for(i=0; i<SPLIT; i++){
      norm_val += norm_ptr[i];
    }
    norm_val = sqrt(norm_val);
  }
  //one norm
  else if(which == 2){
    for(i = 0; i<m; i++){
      if(norm_ptr[i] > norm_val){
        norm_val = norm_ptr[i];
      }
    }
  }
  //one norm
  else{
    for(i = 0; i<n; i++){
      if(norm_ptr[i] > norm_val){
        norm_val = norm_ptr[i];
      }
    }
  }

  cudaFree(M_d);
  free(norm_ptr);

  return norm_val;
}
