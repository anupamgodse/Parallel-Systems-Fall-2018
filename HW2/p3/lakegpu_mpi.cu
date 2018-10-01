/*

Single Author info:

vphadke Vandan V Phadke

Group info:

angodse Anupam N Godse
yjkamdar Yash J Kamdar

*/
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <mpi.h>

#define __DEBUG

#define VSQR 0.1
#define TSCALE 1.0

#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

#define TAG_UO 0
#define TAG_UC 1

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
__global__ void evolve9ptgpu(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t, int rank){

  /*Calculate the offset address for each GPU. Offset will divide the array into 4 even strips*/
  int offset = (rank * (n / 4) * n );

  /*Calculate the index of the current grid point calculation*/
  int idx = offset + blockIdx.x * blockDim.x + threadIdx.x;
  int totalLength = n*n/4;


  if (idx >= 0 && idx < totalLength*4) {

    /*Boudary conditions for the grid*/
    if((idx % n == 0) || ((idx + 1) % n == 0) || idx < n || idx > n*(n-1) - 1)
    {
      un[idx] = 0;
    }
   else
   {
    /*Calculate grid point value using the 9-point scale*/
     un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] +
                  uc[idx + n] + uc[idx - n] + 0.25 * (uc[idx -n - 1] + uc[idx - n + 1] +
                    uc[idx -1 + n] + uc[idx + 1 + n]) - 5 * uc[idx])/(h * h)
                    + (-1 * __expf(-TSCALE * t) * pebbles[idx]));
   }
  }
}

/*Entry point to the GPU computations for the lake problem*/
void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads, int rank, int nprocs)
{

	cudaEvent_t kstart, kstop;
	float ktime;

	/* HW2: Define your local variables here */
  int nBlocks = n / nthreads;
  double t, dt;
  int status;
  MPI_Status mpi_status;

  double *uc, *uo;
  double *un_d, *uc_d, *uo_d, *pebbles_d;

  /*Allocate memory to the temp grids used for computations*/
  uc = (double*)malloc(sizeof(double) * n * n);
  uo = (double*)malloc(sizeof(double) * n * n);

  t = 0.;
  dt = h / 2.;

  /*Copy the grid states of the initial pebble generated grids to the temp grids uo and uc*/
  memcpy(uo, u0, sizeof(double) * n * n);
  memcpy(uc, u1, sizeof(double) * n * n);

  /* Set up device timers */
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

	/*CUDA kernel call preperation code here */
  cudaMalloc((void **) &un_d, sizeof(double) * n * n);
  cudaMalloc((void **) &uc_d, sizeof(double) * n * n);
  cudaMalloc((void **) &uo_d, sizeof(double) * n * n);
  cudaMalloc((void **) &pebbles_d, sizeof(double) * n * n);

  /*Copy host pebbles grid to device(kernel) pebbles grid*/
  cudaMemcpy(pebbles_d, pebbles, sizeof(double) * n * n, cudaMemcpyHostToDevice);

	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));

	/*Main lake simulation loop here */
  while(1)
  {

    /*MPI Send/Receive calls for nodes with rank other than the first node*/
    if(rank != 0) {

      //Receive uo from rank-1.
      status = MPI_Recv(uo+((rank * n * (n / nprocs)) - n), n, MPI_DOUBLE, rank - 1, TAG_UO, MPI_COMM_WORLD, &mpi_status);
      if(status != MPI_SUCCESS) {
        printf("uo receive failed\n");
        fflush(stdout);
        exit(1);
      }

      //Receive uc from rank-1
      status = MPI_Recv(uc+((rank * n * (n / nprocs)) - n), n, MPI_DOUBLE, rank - 1, TAG_UC, MPI_COMM_WORLD, &mpi_status);
      if(status != MPI_SUCCESS) {
        printf("uc receive failed\n");
        fflush(stdout);
        exit(1);
      }

      //Send uo to rank-1
      status = MPI_Send(uo + ((rank * (n / nprocs)) * n), n, MPI_DOUBLE, rank - 1, TAG_UO, MPI_COMM_WORLD);
      if(status != MPI_SUCCESS) {
        printf("uo send to rank -1 failed\n");
        fflush(stdout);
        exit(1);
      }

      //Send uc to rank - 1
      status = MPI_Send(uc+ ((rank * (n / nprocs)) * n), n, MPI_DOUBLE, rank - 1, TAG_UC, MPI_COMM_WORLD);
      if(status != MPI_SUCCESS) {
        printf("uc send to rank -1 failed\n");
        fflush(stdout);
        exit(1);
      }
    }

  /*MPI Send/Receive calls for nodes with rank other than the final node*/
    if(rank != nprocs - 1) {

      //Send u0 to rank + 1
      status = MPI_Send(uo+((((rank + 1) * (n / nprocs)) - 1) * n), n, MPI_DOUBLE, rank + 1, TAG_UO, MPI_COMM_WORLD);
      if(status != MPI_SUCCESS) {
        printf("uo send failed\n");
        fflush(stdout);
        exit(1);
      }

      //Send uc to rank + 1
      status = MPI_Send(uc+((((rank + 1) * (n / nprocs)) - 1) * n), n, MPI_DOUBLE, rank + 1, TAG_UC, MPI_COMM_WORLD);
      if(status != MPI_SUCCESS) {
        printf("uc send failed\n");
        fflush(stdout);
        exit(1);
      }

      //Receive uo from rank+1
      status = MPI_Recv(uo + (((rank+1) * (n / nprocs)) * n), n, MPI_DOUBLE, rank + 1, TAG_UO, MPI_COMM_WORLD, &mpi_status);
      if(status != MPI_SUCCESS) {
        printf("uo send to rank + 1 failed\n");
        fflush(stdout);
        exit(1);
      }

      //Receive uc from rank+1
      status = MPI_Recv(uc+ (((rank+1) * (n / nprocs)) * n), n, MPI_DOUBLE, rank + 1, TAG_UC, MPI_COMM_WORLD, &mpi_status);
      if(status != MPI_SUCCESS) {
        printf("uc send to rank + 1 failed\n");
        fflush(stdout);
        exit(1);
      }

    }

    /*Copy uo and uc from host to device */
    cudaMemcpy(uo_d, uo, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(uc_d, uc, sizeof(double) * n * n, cudaMemcpyHostToDevice);

    /*Main CUDA kernel call for the grid value computations.
    Divided the grid into nprocs(4 in our case) parts using blocks/nprocs as each GPU will calculate only one fourth of the final grid */
    evolve9ptgpu<<<nBlocks*nBlocks / nprocs, nthreads*nthreads>>>(un_d, uc_d, uo_d, pebbles_d, n, h, dt, t, rank);

    /*Copy uo and uc from device to host*/
    cudaMemcpy(uo, uc_d, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(uc, un_d, sizeof(double) * n * n, cudaMemcpyDeviceToHost);

    /*Exit the loop when the lake end_time has reached*/
    if(!tpdt(&t,dt,end_time)) break;
  }

  /*Gather, to combine the grid sections from all 4 nodes into the root node to have one final grid*/
  MPI_Gather(uc + (rank * n * n/4), n*n/4, MPI_DOUBLE, u + (rank * n * n/4),
            n*n/4, MPI_DOUBLE, 0, MPI_COMM_WORLD);


  /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation for rank %d: %f msec\n", rank, ktime);fflush(stdout);

	/*CUDA kernel call processing and cleanup here */
  free(uc);
  free(uo);
  cudaFree(un_d);
  cudaFree(uc_d);
  cudaFree(uo_d);

	/* timer cleanup */
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
}
