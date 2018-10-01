/*

Single Author info:


yjkamdar Yash J Kamdar

Group info:

vphadke Vandan V Phadke
angodse Anupam N Godse

*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

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

/*Evolves the grid using a 5-point scale*/
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);

/*Evolves the grid using a 9-point scale*/
void evolve9pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);

/*Checks whether the evolution time of the grid has reached the end_time provided*/
int tpdt(double *t, double dt, double end_time);

double f(double p, double t);

/*Print the current state of the grid. Called during the start and end*/
void print_heatmap(const char *filename, double *u, int n, double h);

/*Initializes the pebbles. Generates random points on the grid, the pebble will be dropped at.
Also randomly generates the size of the pebble to measure the intensity*/
void init_pebbles(double *p, int pn, int n);

/*Run the grid computation for the final state on the CPU*/
void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time);

/*Run the grid computation for the final state on the GPU*/
extern void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads);

/*Entry point of the program*/
int main(int argc, char *argv[])
{
  /*Int to check if we need to compute through GPU or CPU*/
  int cpu = 0;

  /*Check if nthreads as a perameter is passed*/
  if(argc == 4) {
    cpu = 1;
  }
  else if(argc != 5)
  {
    printf("Usage: %s npoints npebs time_finish nthreads \n",argv[0]);
    return 0;
  }

  /*Variable declarations start*/
  int     npoints   = atoi(argv[1]);
  int     npebs     = atoi(argv[2]);
  double  end_time  = (double)atof(argv[3]);
  int     nthreads;
  int 	  narea	    = npoints * npoints;

  if(cpu == 0)
    nthreads  = atoi(argv[4]);

  double *u_i0, *u_i1;
  double *u_cpu, *u_gpu, *pebs;
  double h;

  double elapsed_cpu, elapsed_gpu;
  struct timeval cpu_start, cpu_end, gpu_start, gpu_end;
  /*Variable declarations end*/

  /*Allocate memory to initial states of the grids*/
  u_i0 = (double*)malloc(sizeof(double) * narea);
  u_i1 = (double*)malloc(sizeof(double) * narea);
  pebs = (double*)malloc(sizeof(double) * narea);

  /*Allocate memory to the GPU/CPU grid to be generated*/
  u_cpu = (double*)malloc(sizeof(double) * narea);
  u_gpu = (double*)malloc(sizeof(double) * narea);

  h = (XMAX - XMIN)/npoints;

  /*Initialize the grids and pebbles positions*/
  init_pebbles(pebs, npebs, npoints);

  /*Generate initial grid point values based on the pebble points generated*/
  init(u_i0, pebs, npoints);
  init(u_i1, pebs, npoints);

  /*Print the heatmap of the initial grid state*/
  print_heatmap("lake_i.dat", u_i0, npoints, h);

  /*CPU computation*/
  if(cpu == 1) {
    gettimeofday(&cpu_start, NULL);
    run_cpu(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time);
    gettimeofday(&cpu_end, NULL);

    /*Calculate time elapsed*/
    elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6)-(
                    cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
    printf("CPU took %f seconds\n", elapsed_cpu);

    print_heatmap("lake_f_cpu.dat", u_cpu, npoints, h);
  }
  /*GPU computation*/
  else {
    gettimeofday(&gpu_start, NULL);
    run_gpu(u_gpu, u_i0, u_i1, pebs, npoints, h, end_time, nthreads);
    gettimeofday(&gpu_end, NULL);
    elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6)-(
                    gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));
    printf("GPU took %f seconds\n", elapsed_gpu);

    print_heatmap("lake_f_gpu.dat", u_gpu, npoints, h);
  }

  /*Free variables*/
  free(u_i0);
  free(u_i1);
  free(pebs);
  free(u_cpu);
  free(u_gpu);

  return 1;
}

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time)
{
  double *un, *uc, *uo;
  double t, dt;

  un = (double*)malloc(sizeof(double) * n * n);
  uc = (double*)malloc(sizeof(double) * n * n);
  uo = (double*)malloc(sizeof(double) * n * n);

  memcpy(uo, u0, sizeof(double) * n * n);
  memcpy(uc, u1, sizeof(double) * n * n);

  t = 0.;
  dt = h / 2.;

  while(1)
  {
    evolve9pt(un, uc, uo, pebbles, n, h, dt, t);

    memcpy(uo, uc, sizeof(double) * n * n);
    memcpy(uc, un, sizeof(double) * n * n);

    if(!tpdt(&t,dt,end_time)) break;
  }

  memcpy(u, un, sizeof(double) * n * n);
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

void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
{
  int i, j, idx;

  for( i = 0; i < n; i++)
  {
    for( j = 0; j < n; j++)
    {
      idx = j + i * n;

      if( i == 0 || i == n - 1 || j == 0 || j == n - 1)
      {
        un[idx] = 0.;
      }
      else
      {
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] +
                    uc[idx + n] + uc[idx - n] - 4 * uc[idx])/(h * h) + f(pebbles[idx],t));
      }
    }
  }
}

void evolve9pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
{
  int i, j, idx;

  for( i = 0; i < n; i++)
  {
    for( j = 0; j < n; j++)
    {
      idx = j + i * n;

      if( i == 0 || i == n - 1 || j == 0 || j == n - 1)
      {
        un[idx] = 0.;
      }
      else
      {
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] +
                    uc[idx + n] + uc[idx - n] + 0.25 * (uc[idx -n - 1] + uc[idx - n + 1] +
                      uc[idx -1 + n] + uc[idx + 1 + n]) - 5 * uc[idx])/(h * h) + f(pebbles[idx],t));
      }
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
