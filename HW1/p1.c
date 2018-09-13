/**********************************************************
 * File p1.c Anupam Godse angodse@ncsu.edu
 * Description: Parallel Systems - CSC 548 Fall 2018 HW1-2
 *  Write an MPI program that determines the point-to-point 
 *  message latency for pairs of nodes. We will exchange
 *  messages with varying message sizes from 32B to 2M 
 *  (32B, 64B, 128B, ..., 2M). The same program should 
 *  iterate through each of the message sizes with 10 
 *  iterations for each. We will use 8 nodes (4 pairs). 
 *  Remember to promptly release the nodes by exiting the 
 *  compute node IMMEDIATELY after your code is finished or 
 *  by using batch scripts.
 *  *******************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include <sys/time.h>
#include <time.h>
/*Total 10 iterations for each message size*/
#define ITERATIONS 10 


int main(int argc, char *argv[]) {
	int size;						//message size
	int nbProcesses;				//total processes
	int processRank;				//self procesRank
	double startTime;				//message sent time
	double endTime;					//message recived after round trip time
	double rtt[ITERATIONS];			//round trip time for each ITERATION 
	double totalRtt;				//total round trip time for ITERATIONS
	double meanRtt;					//mean round trip time
	double sd;						//standard deviation
	int source;						//message source(processRank)
	int destination;				//message destinatio(processRank);
	int tag=0;						//message tag 
	MPI_Status status;				//message receive status
	int mpi_ret;

	MPI_Init(&argc, &argv);			//MPI initialization
	MPI_Comm_size(MPI_COMM_WORLD, &nbProcesses);	//total number of processes
	MPI_Comm_rank(MPI_COMM_WORLD, &processRank);	//getting rank of self

	char processor_name[8];
	int resultlen;
		
	MPI_Get_processor_name(processor_name, &resultlen);
	
	double meanCollected[8];
	double sdCollected[8];
	
	int gather_process_rank = 0;
	int k = 0;									//loop variable	
	for (size=32; size<=2097152; size*=2) {				
		
		//only one pair initiates a round trip	
		if(processRank%2==0) {
			struct timeval tv1, tv2;
			int message[2097152/sizeof(int)];
			
			/*destination = to send message to another process in that pair*/ 
			destination = processRank + 1;
			
			/*source = to receive message from another process in that pair*/
			source = processRank + 1;
			
			int i;		
			totalRtt =  0; 
			for(i = 0; i < ITERATIONS; i++) {
				
				/*round trip time starts*/
				int e = gettimeofday(&tv1, NULL);
				if(e == -1) {
					printf("gettimeofday() failed\n");
					exit(1);
				}				
				
				/*sending message*/
				mpi_ret = MPI_Send(message, size/sizeof(int), MPI_INT, destination,
					tag, MPI_COMM_WORLD);
				if(mpi_ret != MPI_SUCCESS) {
					printf("MPI sending error\n");
					exit(1);
				}
					
				
				/*Waiting for message to be received*/ 
				mpi_ret = MPI_Recv(message, size/sizeof(int), MPI_INT, source, tag,
					 MPI_COMM_WORLD, &status);
				if(mpi_ret != MPI_SUCCESS) {
					printf("MPI receiving error\n");
					exit(1);
				}
				
				e = gettimeofday(&tv2, NULL);
				if(e == -1) {
					printf("gettimeofday() failed\n");
					exit(1);
				}				
				
				rtt[i] = (tv2.tv_sec - tv1.tv_sec) + ((tv2.tv_usec - tv1.tv_usec)/1000000.0);
			
				/*excluding first message exchange*/
				if(i != 0) {
					totalRtt += rtt[i]; //in seconds
				}/*endif*/
			
			}/*for loop for ITERATIONS end*/ 
			
			/*calculating mean rtt*/
			meanRtt = totalRtt / (ITERATIONS - 1);
			
			/*standard deviation*/
			int s;
			double temp = 0;
			for(s = 1; s < ITERATIONS; s++) {
				temp += pow(rtt[s] - meanRtt, 2);
			}/*endfor*/
			temp /= ITERATIONS - 1;
			sd = sqrt(temp); 
		
			/*root process gathers mean rtt from other pairs*/	
			if(MPI_Gather(&meanRtt, 1, MPI_DOUBLE, meanCollected, 1, MPI_DOUBLE, gather_process_rank, MPI_COMM_WORLD)==-1) {
				printf("Gather Failed\n");
				exit(1);
			}/*endif*/
			
			/*root process gathers standard deviation values from other pairs*/	
			if(MPI_Gather(&sd, 1, MPI_DOUBLE, sdCollected, 1, MPI_DOUBLE, gather_process_rank, MPI_COMM_WORLD)==-1) {
				printf("Gather Failed\n");
				exit(1);
			}/*endif*/
			
		}/*endif*/ 
		else {/*second process per pair*/ 
			
			/*source and destination is first process per pair*/
			source = processRank - 1; 
			destination = processRank - 1;
			
			double dummy;
			int message[2097152/sizeof(int)];
			
			/*receive and send message from and to the first process*/
			int i; 
			for(i = 0; i < ITERATIONS; i++) { 
				MPI_Recv(message, size/sizeof(int), MPI_INT, source, tag,
					MPI_COMM_WORLD, &status); 
			 
				MPI_Send(message, size/sizeof(int), MPI_INT, destination,
					 tag, MPI_COMM_WORLD);
			}/*for loop ends*/
			
			/*Sending dummy values to root to overcome stucking of root waiting for these values*/
			if(MPI_Gather(&dummy, 1, MPI_DOUBLE, meanCollected, 1, MPI_DOUBLE, gather_process_rank, MPI_COMM_WORLD)==-1) {
				printf("Gather Failed\n");
				exit(1);
			}
			if(MPI_Gather(&dummy, 1, MPI_DOUBLE, sdCollected, 1, MPI_DOUBLE, gather_process_rank, MPI_COMM_WORLD)==-1) {
				printf("Gather Failed\n");
				exit(1);
			}
		}/*end of else block*/
		
		/*writing mean rtt and sd gathered for size into the file*/
		if(processRank==gather_process_rank) {
			FILE *fp = NULL;
			if(size == 32) {
				fp = fopen("output.txt", "w+");
				if(fp == NULL) {
					printf("File Open Failed\n");
					exit(1);
				}
				fprintf(fp, "Size\tPair\tAvg. RTT\tStd. Deviation\n");
			}
			else{
				fp = fopen("output.txt", "a");
			}
			int i, pair;
			for(i=0; i<nbProcesses; i+=2) {
					pair = i/2 + 1;	
					fprintf(fp, "%d\t%d\t%.14lf\t%.14lf\n", size, pair, 
						meanCollected[i], sdCollected[i]);
					fflush(fp);
			}	
		}
		k++;
		
	}/*for loop for various sizes ends*/
	
	/*MPI cleanup*/
	MPI_Finalize();
	return 0;
}
