#include<stdio.h>	
#include<stdlib.h>
#include<string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <unistd.h>	
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>		


#define BACKLOG 128
#define MPI_COMM_WORLD 0
#define MPI_DOUBLE 0
#define MPI_CHAR 1
#define MPI_INT 2
#define MPI_SUCCESS 0

//#define GATHER 1
//#define PRINT 1


typedef int MPI_Comm;
typedef int MPI_Datatype;
//typedef int MPI_Status;
typedef struct MPI_Status {
     int count;
     int cancelled;
     int MPI_SOURCE;
     int MPI_TAG;
     int MPI_ERROR;
     
} MPI_Status;	

typedef struct address{
	unsigned long s_addr;
	unsigned short port;
	
}address;

//mpi init
int MPI_Init(int *argc, char ***argv);

//mpi comm size
int MPI_Comm_size(MPI_Comm comm, int *size);

//mpi rank
int MPI_Comm_rank(MPI_Comm comm, int *rank);

//processor name
int MPI_Get_processor_name(char *name, int *resultlen);

//mpi send
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);


//mpi_recv
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);

//mpi gather
int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, 
	int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
//MPI_Barrier
int MPI_Barrier(MPI_Comm comm);

int MPI_Finalize();
