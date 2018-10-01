/*

Single Author info:

vphadke Vandan V Phadke

Group info:

angodse Anupam N Godse
yjkamdar Yash J Kamdar

*/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

/*Initializes the initial grid point values using the generated pebble positions*/
void init(double *u, double *pebbles, int n);

/*Checks whether the evolution time of the grid has reached the end_time provided*/
int tpdt(double *t, double dt, double end_time);

/*Print the current state of the grid. Called during the start and end*/
void print_heatmap(const char *filename, double *u, int n, double h);

double f(double p, double t);

/*Initializes the pebbles. Generates random points on the grid, the pebble will be dropped at.
Also randomly generates the size of the pebble to measure the intensity*/
void init_pebbles(double *p, int pn, int n);

/*Run the grid computation for the final state on the GPU*/
extern void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads, int rank, int nprocs);


/*Entry point of the program*/
int main(int argc, char *argv[])
{

  /*Initialize the MPI for processing on multiple nodes*/
  int a = MPI_Init(NULL, NULL);
  if(a != MPI_SUCCESS) {
    printf("MPI Init failed\n");
    fflush(stdout);
  }

  /*Check the command line arguments provided*/
  if(argc != 5)
  {
    printf("Usage: %s npoints npebs time_finish nthreads \n",argv[0]);
    return 0;
  }

  /*Variable declarations start*/
  int     npoints   = atoi(argv[1]);
  int     npebs     = atoi(argv[2]);
  double  end_time  = (double)atof(argv[3]);
  int     nthreads  = atoi(argv[4]);
  int 	  narea	    = npoints * npoints;
  int     rank      ;
  int     nprocs;

  double *u_i0, *u_i1;
  double *u_gpu, *pebs;
  double h;

  double elapsed_gpu;
  struct timeval gpu_start, gpu_end;

  int status;
  /*Variable declarations end*/

  /*Allocate memory to initial states of the grids*/
  u_i0 = (double*)malloc(sizeof(double) * narea);
  u_i1 = (double*)malloc(sizeof(double) * narea);
  pebs = (double*)malloc(sizeof(double) * narea);

  /*Allocate memory to the GPU grid to be generated*/
  u_gpu = (double*)malloc(sizeof(double) * narea);

  /*Fetch the number of processors using MPI*/
  status = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if(status != MPI_SUCCESS) {
    printf("MPI Size failed\n");
    fflush(stdout);
  }

  /*Fetch the rank of the current processor using MPI*/
  status = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(status != MPI_SUCCESS) {
    printf("MPI Rank failed\n");
    fflush(stdout);
  }



  h = (XMAX - XMIN)/npoints;


  if (rank == 0) {
    /*Initialize the grid only on rank 0*/

    printf("Running %s with (%d x %d) grid, until %f, with %d threads\n", argv[0], npoints, npoints, end_time, nthreads);

    /*Initialize the pebble grid locations and sizes*/
    init_pebbles(pebs, npebs, npoints);

    /*Generate initial grid point values based on the pebble points generated*/
    init(u_i0, pebs, npoints);
    init(u_i1, pebs, npoints);

    /*Print the heatmap of the initial grid state*/
    print_heatmap("lake_i.dat", u_i0, npoints, h);

    /*Start GPU timer*/
    gettimeofday(&gpu_start, NULL);

  }

  //Broadcast the initial states from rank 0 to all other processor ranks
  MPI_Bcast(pebs, narea, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(u_i0, narea, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(u_i1, narea, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  run_gpu(u_gpu, u_i0, u_i1, pebs, npoints, h, end_time, nthreads, rank, nprocs);


  if (rank == 0) {
    /*End GPU timer*/
    gettimeofday(&gpu_end, NULL);
    elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6)-(
                    gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));

    /*Print the time taken for the GPU computations*/
    printf("GPU took %f seconds\n", elapsed_gpu);
    char filename[128];
    snprintf(filename, 128, "lake_f_gpu.dat");

    /*Print the heatmap of the grid generated using GPU*/
    print_heatmap(filename, u_gpu, npoints, h);
  }

  /*Free the allocated memories*/
  free(u_i0);
  free(u_i1);
  free(pebs);
  free(u_gpu);

  /*Close the MPI connections between nodes*/
  MPI_Finalize();
}

void init_pebbles(double *p, int pn, int n)
{
  int i, j, k, idx;
  int sz;

  srand( time(NULL) );
  memset(p, 0, sizeof(double) * n * n);

  for( k = 0; k < pn ; k++ )
  {
    i = rand() % (n - 4) + 2;
    j = rand() % (n - 4) + 2;
    sz = rand() % MAX_PSZ;
    idx = j + i * n;
    p[idx] = (double) sz;
  }
}

double f(double p, double t)
{
  return -expf(-TSCALE * t) * p;
}

void init(double *u, double *pebbles, int n)
{
  int i, j, idx;

  for(i = 0; i < n ; i++)
  {
    for(j = 0; j < n ; j++)
    {
      idx = j + i * n;
      u[idx] = f(pebbles[idx], 0.0);
    }
  }
}

void print_heatmap(const char *filename, double *u, int n, double h)
{
  int i, j, idx;

  FILE *fp = fopen(filename, "w");

  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      idx = j + i * n;
      fprintf(fp, "%f %f %f\n", i*h, j*h, u[idx]);
    }
  }

  fclose(fp);
}
