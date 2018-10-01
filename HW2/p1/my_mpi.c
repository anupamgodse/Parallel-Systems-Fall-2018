#include "my_mpi.h"


int nb_processes;
int my_rank;
char my_name[32];
int my_sock;

address *all_addr;


int MPI_Barrier(MPI_Comm comm) {
	int tag = -1;
	MPI_Status mpi_status;
	
	if(my_rank == 0) {
		int i, status;
		double check;
		//0 wating to see if everyone has reached barrier
		for(i = 1; i < nb_processes; i++) {
			status = MPI_Recv(&check, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &mpi_status);
           	if(status != MPI_SUCCESS) {
				#ifdef PRINT
            	printf("Error in receiving data\n");
				fflush(stdout);
				#endif
                return -1;        
  		  	}
			#ifdef PRINT
			printf("At barrier received from %d\n", mpi_status.MPI_SOURCE);
			fflush(stdout);
			#endif
		}
		check = 1.0;
		//sending ack to everyone 
		#ifdef PRINT
		printf("At barrier sending ack from %d to %d\n", 0, i);
		fflush(stdout);
		#endif
		for(i = 1; i < nb_processes; i++) {
			status = MPI_Send(&check, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
			if (status != MPI_SUCCESS) {
				printf("Error in sending data\n");
				fflush(stdout);
				return -1;        
			}
		#ifdef PRINT
		printf("At barrier sent ack from %d to %d\n", 0, i);
		fflush(stdout);
		#endif	
		}
	}	
	else {
		double check;
		int status;
		//send to 0 that my_rank has reached barrier
		#ifdef PRINT
		printf("At barrier sending from %d\n", my_rank);
		fflush(stdout);
		#endif	
		status = MPI_Send(&check, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
		if (status != MPI_SUCCESS) {
			printf("Error in sending data\n");
			fflush(stdout);
			return -1;        
		}
		#ifdef PRINT
		printf("At barrier sent from %d\n", my_rank);
		fflush(stdout);
		#endif	
		
		//wait to receive ack from 0
		#ifdef PRINT
		printf("At barrier receiving ack at  %d\n", my_rank);
		fflush(stdout);
		#endif	
		status = MPI_Recv(&check, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &mpi_status);
		if(status != MPI_SUCCESS) {
			printf("Error in receiving data\n");
			fflush(stdout);
			return -1;        
		}
		#ifdef PRINT
		printf("At barrier received ack by  %d\n", my_rank);
		fflush(stdout);
		#endif	
	}
	return MPI_SUCCESS;
}


int MPI_Finalize() {
	if(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS) {
		close(my_sock);
		return MPI_SUCCESS;
	}
	else {
		return -1;
	}
}

//mpi init
int MPI_Init(int *argc, char ***argv) {
	
	
	//declarations
	char hostname[255]; //same as my_name
	struct hostent *host_entry;
	struct sockaddr_in sock_addr;
	socklen_t sock_addr_len = sizeof(sock_addr);	

	//get processor_name, rank and total number of processes
	strcpy(my_name, (*argv)[1]);
	my_rank = atoi((*argv)[2]);
	nb_processes = atoi((*argv)[3]);


	//all has ip an port for all processes	
	all_addr = (address *)malloc(sizeof(address) * nb_processes); 
	
	//create socket file descriptor
	if((my_sock = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
		printf("create socket error: %s\n", strerror(errno));
		fflush(stdout);
		return -1;
	}

	#ifdef PRINT
	printf("Socket created by %d\n", my_rank);
	fflush(stdout);
	#endif

	//set address fields
    sock_addr.sin_family = AF_INET;
	sock_addr.sin_addr.s_addr = INADDR_ANY;
    sock_addr.sin_port = htons(0); //0 for random unused port	
	
	//bind to random unsed port 
	if(bind(my_sock, (struct sockaddr *)(&sock_addr), sock_addr_len) == -1) {
		printf("binding error: %s\n", strerror(errno));
		fflush(stdout);
		return -1;
	}
	
	#ifdef PRINT
	printf("Binded by %d\n", my_rank);
	fflush(stdout);
	#endif

	//get assigned port number
	if(getsockname(my_sock, (struct sockaddr *)(&sock_addr), &sock_addr_len) == -1) {
		printf("getsockname error = %s\n", strerror(errno));
		fflush(stdout);
		return -1;
	}

	if((listen(my_sock, BACKLOG)) == -1) {
		printf("listen failure: %s\n", strerror(errno));
		fflush(stdout);
		return -1;
	}
	#ifdef PRINT
	printf("got sock name by %d\n", my_rank);
	fflush(stdout);
	#endif

	unsigned short port = sock_addr.sin_port;
	
	#ifdef PRINT
	printf("myrank = %d\t port = %d\t short port = %d\n", my_rank, sock_addr.sin_port, port);
	fflush(stdout);
	#endif

	//gethostname
	gethostname(hostname, 254);
	
	#ifdef PRINT
	printf("Got hostname by %d\n", my_rank);
	fflush(stdout);
	#endif
	//get ip_address
	host_entry = gethostbyname(hostname);
	if(!host_entry) {
		printf("gethostbyname error: %s\n", strerror(errno));
		fflush(stdout);
		return -1;
	}
	
	#ifdef PRINT
	printf("Got hostByname by %d\n", my_rank);
	fflush(stdout);
	#endif

	//open a file to write address and port
	char filename[4];
	snprintf(filename, 4, "%d", my_rank);
	int fd = open(filename, O_WRONLY, 0777);
	if(fd == -1) {
		printf("Open failed: %s\n", strerror(errno));
		fflush(stdout);
		return -1;
	}
	
	if(write(fd, &(sock_addr.sin_port), sizeof(sock_addr.sin_port)) == -1) {
		printf("Port File write error: %s\n", strerror(errno));
		fflush(stdout);
		return -1;
	}
	if(write(fd, (host_entry->h_addr_list)[0], sizeof(sock_addr.sin_addr.s_addr)) == -1) {
		printf("IP File write error: %s\n", strerror(errno));
		fflush(stdout);
		return -1;
	}
	close(fd);
	
	#ifdef PRINT
	printf("Data written by %d\n", my_rank);
	fflush(stdout);
	printf("after \tmyrank = %d\t port = %d\n", my_rank, sock_addr.sin_port);
	fflush(stdout);
	#endif

	/*create a dummy file to indicate other processes of finished writing of above address*/
	snprintf(filename, 4, "w%d", my_rank);
	fd = creat(filename, 0777);
	if(fd == -1) {
		printf("Create failed: %s\n", strerror(errno));
		fflush(stdout);
		return -1;
	}
	
	#ifdef PRINT 
	printf("Dummy file created %d\n", my_rank);
	fflush(stdout);
	#endif
	char check_file[4];
	int i;
	usleep(100000);
	/*wait for other processes to create their own files*/
	for(i = 0; i < nb_processes; i++) {
		#ifdef PRINT
		printf("my rank %d checking for existance of %d\n", my_rank, i);
		fflush(stdout);
		#endif
		snprintf(check_file, 4, "w%d", i);
		while(access(check_file, F_OK) == -1) {
			int temp;
			#ifdef PRINT
			printf("checkfile by %d for %s\n", my_rank, check_file);
			fflush(stdout);
			#endif
			usleep(100000);
		}
		#ifdef PRINT 
		printf("my rank %d check for existance of %d\n", my_rank, i);
		fflush(stdout);
		#endif
	}
		
	//now you are sure that each process has created its own file
	//this indicates that they have finished writing their address
	//to file
	
	//close file and 
	close(fd);
	
	//now as all processes have written their ip and port 
	//into their file read it for all processes
	for(i = 0; i < nb_processes; i++) {
		snprintf(filename, 4, "%d", i);

		int fd = open(filename, O_RDONLY);		
		if(fd == -1) {
			printf("Open failed: %s\n", strerror(errno));
			fflush(stdout);
			return -1;
		}
		if(read(fd, &(all_addr[i].port), sizeof(all_addr[i].port)) == -1) {
			printf("Port File read error: %s\n", strerror(errno));
			fflush(stdout);
			return -1;
		}
		if(read(fd, &(all_addr[i].s_addr), sizeof(all_addr[i].s_addr)) == -1) {
			printf("IP File read error: %s\n", strerror(errno));
			fflush(stdout);
			return -1;
		}
		close(fd);
	}
	//now my_rank knows ip and port for all the processes

	#ifdef PRINT 
	char ip[255];
	if(my_rank == 0) {
			for(i = 0; i < nb_processes; i++) {
					if((inet_ntop(AF_INET, &(all_addr[i].s_addr), ip, 255) == NULL)) {
						printf("inet top error %s\n", strerror(errno));;
						fflush(stdout);
						return -1;
					}
					printf("rank = %d In init : ip = %s \t port = %d \tprocess:%d\n", my_rank, ip,all_addr[i].port, i);
					fflush(stdout);
			}
	}
	#endif
	return MPI_SUCCESS;
}

//mpi comm size
int MPI_Comm_size(MPI_Comm comm, int *size) {
	*size = nb_processes;
	return MPI_SUCCESS;
}

//mpi rank
int MPI_Comm_rank(MPI_Comm comm, int *rank) {
	*rank = my_rank;
	return MPI_SUCCESS;
}

//processor name
int MPI_Get_processor_name(char *name, int *resultlen) {
	strcpy(name, my_name);
	return MPI_SUCCESS;
}

//mpi send
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
	//opening new socket for sending

	int send_sock;
	int written = 0;
	int atonce = 0;

	if((send_sock = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
		printf("create socket error: %s\n", strerror(errno));
		fflush(stdout);
		return -1;
	}
	
	struct sockaddr_in send_sock_addr;
	socklen_t send_sock_addr_len = sizeof(send_sock_addr);	

	//set address fields
    send_sock_addr.sin_family = AF_INET;
	send_sock_addr.sin_addr.s_addr = INADDR_ANY;
    send_sock_addr.sin_port = htons(0); //0 for random unused port	
	
	//bind to random unsed port 
	if(bind(send_sock, (struct sockaddr *)(&send_sock_addr), send_sock_addr_len) == -1) {
		printf("binding error: %s\n", strerror(errno));
		fflush(stdout);
		return -1;
	}
	
	//get assigned port number
	if(getsockname(send_sock, (struct sockaddr *)(&send_sock_addr), &send_sock_addr_len) == -1) {
		printf("getsockname error = %s\n", strerror(errno));
		return -1;
	}
	//opening socket finished

	struct sockaddr_in dest_addr;
	dest_addr.sin_family = AF_INET;
	socklen_t dest_addr_len = sizeof(dest_addr);	

	dest_addr.sin_addr.s_addr = all_addr[dest].s_addr;
	dest_addr.sin_port = all_addr[dest].port;
	
	char ip[255];
	if((inet_ntop(AF_INET, &(dest_addr.sin_addr.s_addr), ip, 255)) == NULL) {
		printf("inet top error %s\n", strerror(errno));;
		fflush(stdout);
        return -1;
    }

	#ifdef PRINT 
	printf("dest = %d at port %d\trank = %d\tip = %s\n", dest, dest_addr.sin_port, my_rank, ip);
	fflush(stdout);
	#endif
	
	//loop until the destination is not listening and the destination address is read rightly
	while((connect(send_sock, (struct sockaddr *) &dest_addr, dest_addr_len)) == -1);
	
	#ifdef PRINT 
	printf("Connected from send in rank %d\n", my_rank);	
	fflush(stdout);
	#endif


	//write to dest socket until write succeeds
  	switch(datatype) {
		case MPI_DOUBLE:
			while(written < count * sizeof(double)) {
					if((atonce = write(send_sock, buf, count * sizeof(double))) == -1) {
						printf("error writing to destination socket %s\n", strerror(errno));
						fflush(stdout);
						return -1;
					}
					else {
						written += atonce;
						#ifdef PRINT
						printf("my rank  = %d \twritten = %d\t count = %d\n", my_rank, written, count * sizeof(double));
                        fflush(stdout);
						#endif
					}
			}
			break;
		case MPI_CHAR:
			while(written < count * sizeof(char)) {
					if((atonce = write(send_sock, buf, count * sizeof(char)) == -1)) {
						printf("error writing to destination socket %s\n", strerror(errno));
						fflush(stdout);
						return -1;
					}
					else {
						written += atonce;
						#ifdef PRINT
						printf("my rank  = %d \twritten = %d\t count = %d\n", my_rank, written, count * sizeof(char));
                        fflush(stdout);
						#endif
					}
			}
			break;
		case MPI_INT:
			while(written < count * sizeof(int)) {
					if((written = write(send_sock, buf, count * sizeof(int))) == -1) {
						printf("error writing to destination socket %s\n", strerror(errno));
						fflush(stdout);
						return -1;
					}
					else {
						written += atonce;
						#ifdef PRINT
						printf("my rank  = %d \twritten = %d\t count = %d\n", my_rank, written, count * sizeof(int));
                        fflush(stdout);
						#endif
					}
			}
			break;
	}
	
	#ifdef PRINT 
	printf("In Send Sent from %d\n", my_rank);
	fflush(stdout);
	#endif
	close(send_sock);
	return MPI_SUCCESS;
}


//mpi_recv
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status) {

	int bytes_received = 0;
	int atonce;
	int from_rank;
	
	//source address (client address)
	struct sockaddr_in src_addr;
	socklen_t src_addr_len = sizeof(src_addr);	
	
	//accept connection from source(client)
	int new_sock_fd = accept(my_sock, (struct sockaddr *)(&src_addr), &src_addr_len);
	if(new_sock_fd < 0) {
		printf("error accepting the source connection %s\n", strerror(errno));
		fflush(stdout);
		return -1;
	}
	
	for(from_rank = 0; from_rank < nb_processes; from_rank++) {
		if(all_addr[from_rank].s_addr == src_addr.sin_addr.s_addr) {
			break;
		}
	}
	(*status).MPI_SOURCE = from_rank;
	(*status).MPI_TAG = tag;
	
	//read data from newely opened socket until send from source
	switch(datatype) {
		case MPI_DOUBLE:
			while(bytes_received < count * sizeof(double)) {
					if((atonce = read(new_sock_fd, buf, sizeof(double) * count)) == -1) {
						printf("Error reading from source socket %s\n", errno);
						fflush(stdout);
						return -1;
					}
					else {
						bytes_received += atonce;
						#ifdef PRINT
						printf("my rank  = %d \treceived = %d\t count = %d\n", my_rank, bytes_received, count * sizeof(double));
						fflush(stdout);
						#endif
					}
			}
			(*status).count = bytes_received;
			break;

		case MPI_CHAR:
			while(bytes_received < count * sizeof(char)) {
					if((atonce = read(new_sock_fd, buf, sizeof(char) * count)) == -1) {
						printf("Error reading from source socket %s\n", errno);
						fflush(stdout);
						return -1;
					}
					else {
						bytes_received += atonce;
						#ifdef PRINT
						printf("my rank  = %d \treceived = %d\t count = %d\n", my_rank, bytes_received, count * sizeof(char));
						fflush(stdout);
						#endif
					}
			}
			(*status).count = bytes_received;
			break;

		case MPI_INT:
			while(bytes_received < count * sizeof(int)) {
					if((atonce = read(new_sock_fd, buf, sizeof(int) * count)) == -1) {
						printf("Error reading from source socket %s\n", errno);
						fflush(stdout);
						return -1;
					}
					else {
						bytes_received += atonce;
						#ifdef PRINT
						printf("my rank  = %d \treceived = %d\t count = %d\n", my_rank, bytes_received, count * sizeof(int));
						fflush(stdout);
						#endif
					}
			}
			(*status).count = bytes_received;
			break;
	
	}
	
	#ifdef PRINT 
	printf("In Recv Received at %d from %d\n", my_rank, (*status).MPI_SOURCE);
	fflush(stdout);
	#endif
	
	close(new_sock_fd);
	return MPI_SUCCESS;
}

//mpi gather
int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, 
	int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {
	
	int status;
	int tag = 0;
	int source;
	int put_from = 0;
	MPI_Status mpi_status;
	void *temp;
	
	//if process root arrives here it creates a file
	char filename[32];
	snprintf(filename, 32, "gather%d", root);
	if(my_rank == root) {
			int fd = creat(filename, 0777);
			if(fd == -1) {
				printf("Create failed: %s\n", strerror(errno));
				fflush(stdout);
				return -1;
			}
	}
	else {//wait for process root to arrrive at gather
			
			usleep(100000);
			//check if root has created a file
			#ifdef PRINT
			printf("my rank %d checking for existance of %d at gather\n", my_rank, root);
			fflush(stdout);
			#endif
			while(access(filename, F_OK) == -1) {
				#ifdef PRINT
				printf("checkfile by %d for %s\n", my_rank, filename);
				fflush(stdout);
				#endif
				usleep(100000);
			}
			#ifdef PRINT 
			printf("my rank %d check for existance of %d at gather\n", my_rank, root);
			fflush(stdout);
			#endif
	}



	//if my rank is root then receive data from all other processes into recv buf
	if(my_rank == root) {
	
		//copy self data into recv buffer
		memcpy(recvbuf+(root*recvcount*sizeof(double)), sendbuf, recvcount * sizeof(double));
		temp = (void *)malloc(recvcount * sizeof(double));

		//recv data for all other processes	
		for(source = 1; source < nb_processes; source++) {
			#ifdef PRINT
			printf("recvbuf = %p\trecvbuf+put_from = %p\tput_from = %d\trecvcount=%d\tsizeof(double) = %d\n",
				recvbuf, recvbuf+put_from, put_from, recvcount, sizeof(double));	
			fflush(stdout);
			printf("In Gather Receiveing at %d\n", my_rank);
			fflush(stdout);
			#endif

			status = MPI_Recv(temp, recvcount, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &mpi_status);
           	if(status != MPI_SUCCESS) {
            	printf("Error in receiving data\n");
				fflush(stdout);
                return -1;        
  		  	}
				
			
			put_from = mpi_status.MPI_SOURCE * recvcount * sizeof(double);
			memcpy(recvbuf+put_from, temp, recvcount * sizeof(double));
			
			#ifdef PRINT
			printf("In Gather Received at %d from %d\t count = %d\t loc = %d\t data = %lf\ttemp = %lf\t put_from = %d\n", 
				my_rank, mpi_status.MPI_SOURCE, mpi_status.count, put_from, *((double *)(recvbuf+put_from)),*( (double *)temp), put_from);
			fflush(stdout);
			#endif
		}

		double ack = 1.0;
	
		for(source = 1; source < nb_processes; source++) {
			status = MPI_Send(&ack, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD);
			if (status != MPI_SUCCESS) {
				printf("Error in sending data\n");
				fflush(stdout);
				return -1;        
			}
			#ifdef PRINT
			printf("In Gather ack Sent to rank %d\n", source);
			fflush(stdout);
			#endif
		}	

	}
	//if my_rank is not root then send data to root process
	else {
		#ifdef PRINT
		printf("In Gather Sending from rank %d \tsendcount = %d\n", my_rank, sendcount);
		fflush(stdout);
		printf("myrank =%d\tsendvalue = %lf\n", my_rank, *((double *)sendbuf));
		fflush(stdout);
		#endif
	
		status = MPI_Send(sendbuf, sendcount, MPI_DOUBLE, root, tag, MPI_COMM_WORLD);
		if (status != MPI_SUCCESS) {
			printf("Error in sending data\n");
			fflush(stdout);
			return -1;        
		}
		#ifdef PRINT
		printf("In Gather Sent from rank %d\n", my_rank);
		fflush(stdout);
		#endif
			
		double ack = 0.0;

		status = MPI_Recv(&ack, 1, MPI_DOUBLE, root, tag, MPI_COMM_WORLD, &mpi_status);
        if(status != MPI_SUCCESS) {
           	printf("Error in receiving data\n");
			fflush(stdout);
            return -1;        
  	 	}
		#ifdef PRINT
		printf("In Gather ack at rank %d\t from rank %d\t ack = %lf\n", my_rank, mpi_status.MPI_SOURCE, ack);
		fflush(stdout);
		#endif
	}
	usleep(100000);	
	if(my_rank == root) {
		if(remove(filename) == -1) {
			printf("remove of filename %s failed by %d at gather\n", filename, my_rank);
			fflush(stdout);
		}
		
	}
	return MPI_SUCCESS;
}
