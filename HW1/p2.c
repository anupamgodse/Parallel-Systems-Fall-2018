#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/* first grid point */
#define   XI              1.0

/* last grid point */
#define   XF              100.0

/* function declarations */
double     fn(double);
void       print_function_data(int, double*, double*, double*);

int main(int argc, char *argv[]){

    // Initialize MPI
    MPI_Init( &argc, &argv );

    // Get rank of the current process
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank); 

    // Hardcoding gathering process as process with rank 0
    int root = 0;

    double prog_start, prog_end; 

    if (rank == root)
    {
        prog_start = MPI_Wtime();    
    }

    // Grid Size variable
    int NGRID;

    // Blocking/Non-blocking type
    int block_type = 0;

    // Manual/MPI Gather
    int gather_type = 0;

    // Grid size input from stdin
    if(argc > 3){
        NGRID = atoi(argv[1]);
        block_type = atoi(argv[2]);
        gather_type = atoi(argv[3]);
    }
    else 
    {
        printf("Please specify the number of grid points, blocking type and Gather type\n");
        exit(0);
    }

    // Domain of the function
    double *xc = (double *)malloc(sizeof(double)* (NGRID + 2));
    double dx;

    // Store values for fn(x) and its derivative
    double *yc, *dyc;

    // Construct grid points based on number of points
    for (int i = 1; i <= NGRID ; i++)
    {
        xc[i] = XI + (XF - XI) * (double)(i - 1)/(double)(NGRID - 1);
    }    

    //step size and boundary points
    dx = xc[2] - xc[1];
    xc[0] = xc[1] - dx;
    xc[NGRID + 1] = xc[NGRID] + dx;
    
    // Compute values for leftmost and rightmost grid point separately as base case
    double base_case_left = fn(xc[0]);
    double base_case_right = fn(xc[NGRID+1]);

    // Get total number of processes running this program parallely
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Number of points which each process will compute 
    int limit = (int) (ceil((double)NGRID/nprocs));

    // Compute starting and ending indexes for computing processes
    int mystart = rank * limit + 1; 
    int myend = mystart + limit - 1;

    // Handles the case where points are not an uneven split between processes 
    // Gives less number of points to the process with last rank
    if (rank==(nprocs-1))
    {
         myend = NGRID;
         limit = myend - mystart + 1;
    }   

    // MPI return status
    int status; 

    // Status of message received
    MPI_Status mpi_status;

    // Mpi request parameter
    MPI_Request request;

    //allocate memory for function values and derivatives
    yc  =   (double*) malloc((limit) * sizeof(double));
    dyc =   (double*) malloc((limit) * sizeof(double));

    yc[0] = fn(xc[mystart]);
    yc[limit-1] = fn(xc[myend]);

    // Send leftmost function value to rank-1 
    if (rank != 0)
    {   
        if (block_type == 1)
            status = MPI_Isend(&yc[0], 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &request);
        else 
            status = MPI_Send(&yc[0], 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);

        if (status != MPI_SUCCESS)
        {
            printf("Error in sending data\n");
            exit(1);        
        }  
    }

    // Send rightmost function value to rank-1
    if (rank != nprocs - 1)
    {
        if (block_type == 1)
            status = MPI_Isend(&yc[limit-1], 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &request);
        else 
            status = MPI_Send(&yc[limit-1], 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
  
        // Check send message status
        if (status != MPI_SUCCESS)
        {
            printf("Error in sending data\n");
            exit(1);        
        }
    }

    // Compute function values for grid points
    for (int i = 1; i < limit - 1; ++i)
    {
        yc[i] = fn(xc[mystart + i]);
    }

    //compute the derivative using first-order finite differencing
    //
    //  d           f(x + h) - f(x - h)
    // ---- f(x) ~ --------------------
    //  dx                 2 * dx
    //
    for (int i = 1; i < limit - 1; ++i)
    {
        dyc[i] = (yc[i + 1] - yc[i - 1])/(2.0 * dx);
    }

    double left_receive ; 
    double right_receive ; 
    
    MPI_Request request_left, request_right; 

    // Receive function values from lower and higher ranked processes
    if (rank != 0 && rank != (nprocs - 1))
    {

        if (block_type == 1)
            status = MPI_Irecv(&left_receive, 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &request_left);
        else 
            status = MPI_Recv(&left_receive, 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &mpi_status);    
        
        // Check receive message status
        if (status != MPI_SUCCESS)
        {
            printf("Error in receiving data\n");
            exit(1);        
        }

        if (block_type == 1)
            status = MPI_Irecv(&right_receive, 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &request_right);
        else        
            status = MPI_Recv(&right_receive, 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &mpi_status);
        
        // Check receive message status
        if (status != MPI_SUCCESS)
        {
            printf("Error in receiving data\n");
            exit(1);        
        }

        if (block_type == 1){
            status = MPI_Wait(&request_right, &mpi_status);

            // Check wait message status
            if (status != MPI_SUCCESS)
            {
                printf("Error in waiting for data\n");
                exit(1);        
            }

            status = MPI_Wait(&request_right, &mpi_status);    

            // Check wait message status
            if (status != MPI_SUCCESS)
            {
                printf("Error in waiting for data\n");
                exit(1);        
            }
        }
        
    }

    // Handles case for the lowest ranked process
    else if (rank == 0)
    {
        if (block_type == 1)
            status = MPI_Irecv(&right_receive, 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &request_right);
        else
            status = MPI_Recv(&right_receive, 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &mpi_status);

        // Check receive message status
        if (status != MPI_SUCCESS)
        {
            printf("Error in receiving data\n");
            exit(1);        
        }

        // Set leftmost base case
        left_receive = base_case_left;

        if (block_type == 1){
            status = MPI_Wait(&request_right, &mpi_status);
            
            // Check wait message status
            if (status != MPI_SUCCESS)
            {
                printf("Error in waiting for data\n");
                exit(1);        
            }
        }
            
        
    }

    // Handles case for the highest ranked process
    else if (rank == (nprocs - 1))
    {
        if (block_type == 1)
            status = MPI_Irecv(&left_receive, 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &request_left);
        else    
            status = MPI_Recv(&left_receive, 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &mpi_status);

        // Check receive message status
        if (status != MPI_SUCCESS)
        {
            printf("Error in receiving data\n");
            exit(1);        
        }

        // Set rightmost base case
        right_receive = base_case_right;

        if (block_type == 1){
            status = MPI_Wait(&request_left, &mpi_status);

            // Check wait message status
            if (status != MPI_SUCCESS)
            {
                printf("Error in waiting for data\n");
                exit(1);        
            }

        }
    }

    // Compute differential for edge cases using received values
    dyc[0] = (yc[1] - left_receive) / (2.0 * dx); 
    dyc[limit-1] = (right_receive - yc[limit-2]) / (2.0 * dx);

    // Variables to gather computed values from different processes
    double *dyc_final;
    double *yc_final;

    if (rank == root)
    {
        yc_final = (double *)malloc(sizeof(double)* (NGRID + 2));
        dyc_final = (double *)malloc(sizeof(double)* (NGRID + 2));

    }

    // Gather using MPI Derivatives
    if (gather_type == 0)
    {
        // Gather computed values
        status = MPI_Gather(yc, limit, MPI_DOUBLE, yc_final, limit, MPI_DOUBLE, root, MPI_COMM_WORLD);

        // Check gather message status
        if (status != MPI_SUCCESS)
        {
            printf("Error in gathering data\n");
            exit(1);        
        }

        status = MPI_Gather(dyc, limit, MPI_DOUBLE, dyc_final, limit, MPI_DOUBLE, root, MPI_COMM_WORLD);
        
        // Check gather message status
        if (status != MPI_SUCCESS)
        {
            printf("Error in gathering data\n");
            exit(1);        
        }        
    }
    // Implementing gather using MPI Send/Receive
    else {

        // We need to send data from all ranks from all processes other than the root processes   
        if (rank != root)
        {
            // Send data from process to the root process
            status = MPI_Send(yc, limit, MPI_DOUBLE, root, 0, MPI_COMM_WORLD);
            
            // Check send message status
            if (status != MPI_SUCCESS)
            {
                printf("Error in sending data\n");
                exit(1);        
            }

            status = MPI_Send(dyc, limit, MPI_DOUBLE, root, 0, MPI_COMM_WORLD);
            
            // Check send message status
            if (status != MPI_SUCCESS)
            {
                printf("Error in sending data\n");
                exit(1);        
            }

        }
        // We need to receive data from all other processes in the root process
        else
        {   

            // Copy data computed in the root process to the final array
            // where all values are collected 
            for (int i = 0; i < limit; ++i)
            {
                yc_final[i] = yc[i];
                dyc_final[i] = dyc[i];
            }

            // Receive data for each process other than the root
            for (int sender = 1; sender < nprocs; ++sender)
            {

                // Compute starting address where the collected data is to be stored
                int mystart = sender * limit;
                
                // Handling edge case where number of elements can be lesser than limit
                if (sender==(nprocs-1))
                {
                    myend = NGRID;
                    limit = myend - mystart + 1;
                }
                
                // Receive data from all processes
                status = MPI_Recv(yc_final+mystart, limit, MPI_DOUBLE, sender, 0, MPI_COMM_WORLD, &mpi_status);
                
                if (status != MPI_SUCCESS)
                {
                    printf("Error in receiving data\n");
                    exit(1);        
                }
                
                status = MPI_Recv(dyc_final+mystart, limit, MPI_DOUBLE, sender, 0, MPI_COMM_WORLD, &mpi_status);
                if (status != MPI_SUCCESS)
                {
                    printf("Error in receiving data\n");
                    exit(1);        
                }
            }
        }
    }

    // Dump data to a file
    if (rank == root)
    {
        print_function_data(NGRID, (xc+1), yc_final, dyc_final);
        prog_end = MPI_Wtime();    
        printf("Elapsed time is %f\n", prog_end - prog_start);
    }
}

//prints out the function and its derivative to a file
void print_function_data(int np, double *x, double *y, double *dydx)
{
        int   i;

        char filename[1024];
        sprintf(filename, "fn-%d.dat", np);

        FILE *fp = fopen(filename, "w");

        for(i = 0; i < np; i++)
        {
                fprintf(fp, "%f %f %f\n", x[i], y[i], dydx[i]);
        }

        fclose(fp);
}
