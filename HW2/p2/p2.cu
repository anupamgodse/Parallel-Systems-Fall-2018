/* Program to compute Pi using Monte Carlo methods */


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#define SEED 35791246


__global__ void getcount(int *count_dev) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		double x, y, z;

		//init random number seed by taking clock() value
		curandState_t state;
		curand_init(clock(), idx, 0, &state);

		//get random x and y between [0,1]
		x = (double)curand_uniform(&state);
		y = (double)curand_uniform(&state);
		z = x * x + y * y;

		//check if z is inside the circle if yes, then increment the count
		if(z <= 1) {
			count_dev[idx] += 1;
		}
		
}

int main(int argc, char** argv) {
		//local variables
		int niter=0;
		double pi;
		int *count_host;
		int final_count = 0;

		//device_variabls
		int *count_dev;

		niter = atoi(argv[1]);
		int block_size = 512;
		int nb_blocks = niter/block_size;	
		int size = sizeof(int) * niter;
		

		//allocate memory for count on host
		count_host = (int *)malloc(size);
		memset(count_host, 0, size);

		//allcoate memory for count on device
		cudaMalloc((void **) &count_dev, size);

		//copy data from host to device initial counters (all 0)
		cudaMemcpy(count_dev, count_host, size, cudaMemcpyHostToDevice);

		getcount<<<nb_blocks, block_size>>>(count_dev);

		cudaMemcpy(count_host, count_dev, size, cudaMemcpyDeviceToHost);

		for (int i=0; i<niter; i++) {
			//printf("final_count = %d\t", final_count);
			final_count += count_host[i];
		}

		pi=(((double)final_count)/niter)*4;
		printf("# of trials= %d , estimate of pi is %.16f \t\n",niter,pi);

		return 0;
}
