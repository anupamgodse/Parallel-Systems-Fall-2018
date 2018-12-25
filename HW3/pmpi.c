#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>
#include <string.h>

int *send_count;
int *all_send_counts;
int my_rank;
int nb_processes;
int status;

int MPI_Init(int *argc, char ***argv) {
	int status_init = PMPI_Init(argc, argv);

	//init value of my_rank with current process rank
	status = PMPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	if(status != MPI_SUCCESS) {
		printf("Getting rank failure\n");
		exit(1);
	}

	//init value of nb_processes with total processes
	status = PMPI_Comm_size(MPI_COMM_WORLD, &nb_processes);
	if(status != MPI_SUCCESS) {
		printf("Getting rank failure\n");
		exit(1);
	}

	//create an array to hold send count to each process
	send_count = (int *)malloc(sizeof(int) * nb_processes);
	if(send_count == NULL) {
		printf("malloc failed\n");
		exit(1);
	}

	//init send count to each process to 0
	memset(send_count, 0, sizeof(int) * nb_processes);
	
	//create array to collect send counts from all processes at rank 0
	if(my_rank == 0) {
			all_send_counts = (int *)malloc(sizeof(int) * nb_processes * nb_processes);
			if(all_send_counts == NULL) {
				printf("malloc failed\n");
				exit(1);
			}
	}


	return status_init;

}

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
    int tag, MPI_Comm comm) {
	
	//increment send count to destination process
	send_count[dest]++;

	return PMPI_Send(buf, count, datatype, dest, tag, comm);
	
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm, MPI_Request *request) {

	//increment send count to destination process
	send_count[dest]++;

	return PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
}

int MPI_Finalize() {
	//gather send count data from all processes at rank 0
	status = PMPI_Gather(send_count, nb_processes, MPI_INT,
               all_send_counts, nb_processes, MPI_INT,
               0, MPI_COMM_WORLD);
	if(status != MPI_SUCCESS) {
		printf("PMPI_Gather failed\n");
		exit(1);
	}	

	//flush sendcount data to file
	if(my_rank == 0) {
		//open a file to write send count data
		FILE *fp;
		fp = fopen("matrix.data", "w+");	
		if(fp == NULL) {
			printf("fopen failed\n");
			exit(1);
		}
		
		//write send count data to file
		int i, j;
		for(i = 0; i < nb_processes; i++) {
			fprintf(fp, "%d\t", i);
			for(j = 0; j < nb_processes; j++) {
				fprintf(fp, "%d\t", all_send_counts[i*nb_processes + j]);
			}
			fprintf(fp, "\n");
		}
		free(all_send_counts);
	}
	
	free(send_count);
	return PMPI_Finalize();
}
