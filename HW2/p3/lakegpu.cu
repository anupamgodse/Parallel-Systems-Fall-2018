/*

Single Author info:


yjkamdar Yash J Kamdar

Group info:

vphadke Vandan V Phadke
angodse Anupam N Godse

*/
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define __DEBUG

#define VSQR 0.1
#define TSCALE 1.0

#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

/**************************************
* void __cudaSafeCall(cudaError err, const char *file, const int line)
* void __cudaCheckError(const char *file, const int line)
*
* These routines were taken from the GPU Computing SDK
* (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"
**************************************/
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef __DEBUG

#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
              file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
  } while ( 0 );
#pragma warning( pop )
#endif  // __DEBUG
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef __DEBUG
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment if not needed.
    /*err = cudaThreadSynchronize();
    if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }*/
  } while ( 0 );
#pragma warning( pop )
#endif // __DEBUG
  return;
}

int tpdt(double *t, double dt, double tf)
{
  if((*t) + dt > tf) return 0;
  (*t) = (*t) + dt;
  return 1;
}

/*9-point evolution of the grid using the GPU*/
__global__ void evolve9ptgpu(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t){

  /*Calculate the index of the current grid point calculation*/
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int totalLength = n*n;

  /*Boudary conditions for the grid*/
  if (idx >= 0 && idx < totalLength) {
    if((idx % n == 0) || ((idx + 1) % n == 0) || idx < n || idx > n*(n-1) - 1)
    {
      un[idx] = 0;
    }
   /*Calculate grid point value using the 9-point scale*/
   else
   {
     un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] +
                  uc[idx + n] + uc[idx - n] + 0.25 * (uc[idx -n - 1] + uc[idx - n + 1] +
                    uc[idx -1 + n] + uc[idx + 1 + n]) - 5 * uc[idx])/(h * h)
                    + (-1 * __expf(-TSCALE * t) * pebbles[idx]));
   }
  }
}

__global__ void evolvegpu(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int totalLength = n*n;

  if (idx >= 0 && idx < totalLength) {
    if((idx % n == 0) || ((idx + 1) % n == 0) || idx < n || idx > n*(n-1) - 1)
    {
      un[idx] = 0;
    }
   else
   {
     un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] +
                 uc[idx + n] + uc[idx - n] - 4 * uc[idx])/(h * h) +  (__expf(-TSCALE * t) * pebbles[idx]));
   }
  }
}

void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads)
{
	cudaEvent_t kstart, kstop;
	float ktime;

	/* HW2: Define your local variables here */
  int nBlocks = n / nthreads;
  double t, dt;

  double *uc, *uo;
  double *un_d, *uc_d, *uo_d, *pebbles_d;

  uc = (double*)malloc(sizeof(double) * n * n);
  uo = (double*)malloc(sizeof(double) * n * n);

  t = 0.;
  dt = h / 2.;

  memcpy(uo, u0, sizeof(double) * n * n);
  memcpy(uc, u1, sizeof(double) * n * n);

  /* Set up device timers */
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

	/* HW2: Add CUDA kernel call preperation code here */
  cudaMalloc((void **) &un_d, sizeof(double) * n * n);
  cudaMalloc((void **) &uc_d, sizeof(double) * n * n);
  cudaMalloc((void **) &uo_d, sizeof(double) * n * n);
  cudaMalloc((void **) &pebbles_d, sizeof(double) * n * n);

  cudaMemcpy(pebbles_d, pebbles, sizeof(double) * n * n, cudaMemcpyHostToDevice);

	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));

	/* HW2: Add main lake simulation loop here */
  while(1)
  {

    cudaMemcpy(uo_d, uo, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(uc_d, uc, sizeof(double) * n * n, cudaMemcpyHostToDevice);

    evolve9ptgpu<<<nBlocks*nBlocks, nthreads*nthreads>>>(un_d, uc_d, uo_d, pebbles_d, n, h, dt, t);
    //evolvegpu<<<nBlocks*nBlocks, nthreads*nthreads>>>(un_d, uc_d, uo_d, pebbles_d, n, h, dt, t);
    cudaMemcpy(uo, uc_d, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(uc, un_d, sizeof(double) * n * n, cudaMemcpyDeviceToHost);

    if(!tpdt(&t,dt,end_time)) break;
  }

  memcpy(u, uc, sizeof(double) * n * n);

  /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

	/* HW2: Add post CUDA kernel call processing and cleanup here */
  free(uc);
  free(uo);
  cudaFree(un_d);
  cudaFree(uc_d);
  cudaFree(uo_d);

	/* timer cleanup */
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
}
