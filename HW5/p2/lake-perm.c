/*

Single Author info:

vphadke Vandan V Phadke

Group info:

yjkamdar Yash J Kamdar

angodse Anupam N Godse

*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "jemalloc/jemalloc.h"

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

#define BACK_FILE "/tmp/app.back"
#define MMAP_FILE "/tmp/app.mmap"
#define MMAP_SIZE ((size_t)1 << 30)

/*Initializes the initial grid point values using the generated pebble positions*/
void init(double *u, double *pebbles, int n);

/*Evolves the grid using a 5-point scale*/
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);

/*Evolves the grid using a 9-point scale*/
void evolve9pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);

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

/*Checks whether the evolution time of the grid has reached the end_time provided*/
int tpdt(double *t, double dt, double tf)
{
  if((*t) + dt > tf) return 0;
  (*t) = (*t) + dt;
  return 1;
}

PERM double *pebs;
int do_restore ;

/*Entry point of the program*/
int main(int argc, char *argv[])
{

  /* Check for "-r" argument for restoring perm variables*/ 
  do_restore = argc > 1 && strcmp("-r", argv[1]) == 0;
  const char *mode = (do_restore) ? "r+" : "w+";
  
  // Persistent memory initialization
  perm(PERM_START, PERM_SIZE);
  mopen(MMAP_FILE, mode, MMAP_SIZE);
  bopen(BACK_FILE, mode);

  /*Variable declarations start*/
  int     npoints   = 128;
  int     npebs     = 8;
  double  end_time  = 1.0;
  int 	  narea	    = npoints * npoints;

  double *u_i0, *u_i1;
  double *u_cpu; 
  
  double h;

  double elapsed_cpu;
  struct timeval cpu_start, cpu_end, gpu_start, gpu_end;
  /*Variable declarations end*/

  /*Allocate memory to initial states of the grids*/
  u_i0 = (double*)malloc(sizeof(double) * narea);
  u_i1 = (double*)malloc(sizeof(double) * narea);
  pebs = (double*)malloc(sizeof(double) * narea);

  /*Allocate memory to the GPU/CPU grid to be generated*/
  u_cpu = (double*)malloc(sizeof(double) * narea);

  h = (XMAX - XMIN)/npoints;

  
  if (!do_restore)
  {
    /*Initialize the grids and pebbles positions*/
    init_pebbles(pebs, npebs, npoints);
    mflush();
    backup();
  }
  else {
    
    /* Restore all variables*/
    printf("restarting...\n");
    restore();
  }

  
  /*Generate initial grid point values based on the pebble points generated*/
  init(u_i0, pebs, npoints);
  init(u_i1, pebs, npoints);

  /*Print the heatmap of the initial grid state*/
  print_heatmap("lake_i.dat", u_i0, npoints, h);

  gettimeofday(&cpu_start, NULL);
  run_cpu(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time);
  gettimeofday(&cpu_end, NULL);

  /*Calculate time elapsed*/
  elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6)-(
                  cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
  printf("CPU took %f seconds\n", elapsed_cpu);

  print_heatmap("lake_f_cpu.dat", u_cpu, npoints, h);

  /*Free variables*/
  // free(u_i0);
  // free(u_i1);
  // free(pebs);
  // free(u_cpu);

  // Cleanup
  mclose();
  bclose();
  remove(BACK_FILE);
  remove(MMAP_FILE);

  return 0;
}

// Variables to be stored in persistent memory. 
PERM double *uc, *uo;
PERM double t;

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time)
{
  
  double *un;
  double dt;

  // Initialize only when running for the first time. 
  if (!do_restore)
  {

    uc = (double*)malloc(sizeof(double) * n * n);
    uo = (double*)malloc(sizeof(double) * n * n);  

    memcpy(uo, u0, sizeof(double) * n * n);
    memcpy(uc, u1, sizeof(double) * n * n);

    t = 0.;

    // Store global variables
    mflush();
    
    // backup initialized variables
    backup();
  }

  un = (double*)malloc(sizeof(double) * n * n);
  
  
  dt = h / 2.;

  while(1)
  {

    printf("Current time is %f\n", t);

    evolve9pt(un, uc, uo, pebbles, n, h, dt, t);

    memcpy(uo, uc, sizeof(double) * n * n);
    memcpy(uc, un, sizeof(double) * n * n);

    if(!tpdt(&t,dt,end_time)) break;

    // Store persistent variables to memory.
    backup();
  }

  memcpy(u, un, sizeof(double) * n * n);
}

void init_pebbles(double *p, int pn, int n)
{
  int i, j, k, idx;
  int sz;

  srand(1234);
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
