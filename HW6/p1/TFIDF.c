#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<dirent.h>
#include<math.h>

#include<mpi.h>
#define MASTER 0

#define MAX_WORDS_IN_CORPUS 32
#define MAX_FILEPATH_LENGTH 16
#define MAX_WORD_LENGTH 16
#define MAX_DOCUMENT_NAME_LENGTH 8
#define MAX_STRING_LENGTH 64

typedef char word_document_str[MAX_STRING_LENGTH];

typedef struct o {
	char word[32];
	char document[8];
	int wordCount;
	int docSize;
	int numDocs;
	int numDocsWithWord;
	double val;
} obj;

typedef struct w {
	char word[32];
	int numDocsWithWord;
	int currDoc;
} u_w;

static int myCompare (const void * a, const void * b)
{
    return strcmp (a, b);
}

int main(int argc , char *argv[]){
	//MPI declarations
	//get nb ranks
	
	int rank, nb_processes, workers;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nb_processes);

	workers = nb_processes-1;

	DIR* files;
	struct dirent* file;
	int i,j;
	int numDocs = 0, docSize, contains;
	char filename[MAX_FILEPATH_LENGTH], word[MAX_WORD_LENGTH], document[MAX_DOCUMENT_NAME_LENGTH];
	
	// Will hold all TFIDF objects for all documents
	obj TFIDF[MAX_WORDS_IN_CORPUS];
	int TF_idx = 0;
	
	// Will hold all unique words in the corpus and the number of documents with that word
	u_w unique_words[MAX_WORDS_IN_CORPUS];
	int uw_idx = 0;
	
	// Will hold the final strings that will be printed out
	word_document_str strings[MAX_WORDS_IN_CORPUS];
	
	
	
	if(rank == MASTER) {
		//Count numDocs
		if((files = opendir("input")) == NULL){
			printf("Directory failed to open\n");
			exit(1);
		}
		while((file = readdir(files))!= NULL){
			// On linux/Unix we don't want current and parent directories
			if(!strcmp(file->d_name, "."))	 continue;
			if(!strcmp(file->d_name, "..")) continue;
			numDocs++;
		}
	
		int nb_docs_for_each = numDocs/workers, nb_docs_for_last;
        int remainder = numDocs%workers;

        //first remainder workers will get 1 extra file from remaining workers
        if(remainder) {
            nb_docs_for_each++;
        }
        int offset = nb_docs_for_each * remainder;
        for(i=1; i<=remainder; i++) {
            MPI_Send(&nb_docs_for_each, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&numDocs, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

            for(j = 1; j <= nb_docs_for_each; j++) {
                sprintf(document, "doc%d", (i-1)*nb_docs_for_each + j);
                sprintf(filename,"input/%s",document);
                MPI_Send(filename, strlen(filename), MPI_CHAR, i, 0, MPI_COMM_WORLD);
            }
        }
		//remaining workers will get one less file than first remainder workers
        if(remainder) {
            nb_docs_for_each--;
        }
        for(i=1; i<=(workers-remainder); i++) {
            MPI_Send(&nb_docs_for_each, 1, MPI_INT, remainder+i, 0, MPI_COMM_WORLD);
            MPI_Send(&numDocs, 1, MPI_INT, remainder+i, 0, MPI_COMM_WORLD);

            for(j = 1; j <= nb_docs_for_each; j++) {
                sprintf(document, "doc%d", offset + ((i-1)*nb_docs_for_each) + j);
                sprintf(filename,"input/%s",document);
                MPI_Send(filename, strlen(filename), MPI_CHAR, remainder+i, 0, MPI_COMM_WORLD);
            }
        }

		int k, uw_idx_i;
		u_w word_w[MAX_WORDS_IN_CORPUS];
		
		//now master will receive u_w struct from each worker and aggregate them
		for(i=1; i<=workers; i++) {
			MPI_Recv(&uw_idx_i, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(word_w, sizeof(u_w) * uw_idx_i, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			for(j=0; j<uw_idx_i; j++) {
				contains = 0;
				// If unique_words array already contains the word, just increment numDocsWithWord
				for(k=0; k<uw_idx; k++) {
					if(!strcmp(unique_words[k].word, word_w[j].word)){
						contains = 1;
						unique_words[k].numDocsWithWord += word_w[j].numDocsWithWord;
						unique_words[k].currDoc = word_w[j].currDoc;
						break;
					}
				}
			
				// If unique_words array does not contain it, make a new one with numDocsWithWord=word_w.numDocsWithWord
				if(!contains) {
					strcpy(unique_words[uw_idx].word, word_w[j].word);
					unique_words[uw_idx].numDocsWithWord = word_w[j].numDocsWithWord;
					uw_idx++;
				}
			}

		}
		
		//now master will send this unique words array to every worker so that the worker knows
		//how many other documents have the same word it has to update nunDocsWithWord
		MPI_Request request;
		for(i=1; i<=workers; i++) {
			//Isend so that master wont have to wait to send to other workers
			MPI_Send(&uw_idx, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Isend(unique_words, sizeof(u_w) * uw_idx, MPI_BYTE, i, 0, MPI_COMM_WORLD, &request);
		}		


		//receive TFIDF from all the workers
		int TF_idx_i;
		for(i=1 ;i<=workers;i++) {
			MPI_Recv(&TF_idx_i, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(TFIDF+TF_idx, sizeof(obj) * TF_idx_i, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			TF_idx += TF_idx_i;
		}
		
		// Print TF job similar to HW4/HW5 (For debugging purposes)
		printf("-------------TF Job-------------\n");
		for(j=0; j<TF_idx; j++)
			printf("%s@%s\t%d/%d\n", TFIDF[j].word, TFIDF[j].document, TFIDF[j].wordCount, TFIDF[j].docSize);
		
		
		// Print IDF job similar to HW4/HW5 (For debugging purposes)
		printf("------------IDF Job-------------\n");
		for(j=0; j<TF_idx; j++)
			printf("%s@%s\t%d/%d\n", TFIDF[j].word, TFIDF[j].document, TFIDF[j].numDocs, TFIDF[j].numDocsWithWord);
			
		
		// Populate strings array with calulated TFIDF values
		for(j=0; j<TF_idx; j++) {
			sprintf(strings[j], "%s@%s\t%.16f", TFIDF[j].document, TFIDF[j].word, TFIDF[j].val);
		}
			
			
		qsort(strings, TF_idx, sizeof(char)*MAX_STRING_LENGTH, myCompare);
		FILE* fp = fopen("output.txt", "w");
		if(fp == NULL){
			printf("Error Opening File: output.txt\n");
			exit(0);
		}
		for(i=0; i<TF_idx; i++)
			fprintf(fp, "%s\n", strings[i]);
		fclose(fp);
		
		
	}//end of if rank==MASTER
	else {
	
		int totalDocs;
		
		//worker will receive the numDocs it needs to process
		MPI_Recv(&numDocs, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&totalDocs, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
		// Loop through each document and gather TFIDF variables for each word
		for(i=1; i<=numDocs; i++){
			
			int docNameIdx;

			memset(filename, '\0', MAX_FILEPATH_LENGTH);

			MPI_Recv(filename, MAX_FILEPATH_LENGTH, MPI_CHAR, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			for(j=strlen(filename)-1;j>=0;j--)
				if(filename[j] == '/')
					break;
				
			docNameIdx = j+1;			

			FILE* fp = fopen(filename, "r");
			if(fp == NULL){
				printf("Error Opening File: %s\n", filename);
				exit(0);
			}
		
			// Get the document size
			docSize = 0;
			while((fscanf(fp,"%s",word))!= EOF)
				docSize++;
		
			// For each word in the document
			fseek(fp, 0, SEEK_SET);
			while((fscanf(fp,"%s",word))!= EOF){
				contains = 0;
			
				// If TFIDF array already contains the word@document, just increment wordCount and break
				for(j=0; j<TF_idx; j++) {
					if(!strcmp(TFIDF[j].word, word) && !strcmp(TFIDF[j].document, filename+docNameIdx)){
						contains = 1;
						TFIDF[j].wordCount++;
						break;
					}
				}
			
				//If TFIDF array does not contain it, make a new one with wordCount=1
				if(!contains) {
					strcpy(TFIDF[TF_idx].word, word);
					strcpy(TFIDF[TF_idx].document, filename+docNameIdx);
					TFIDF[TF_idx].wordCount = 1;
					TFIDF[TF_idx].docSize = docSize;
					TFIDF[TF_idx].numDocs = totalDocs;
					TF_idx++;
				}
			
				contains = 0;
				// If unique_words array already contains the word, just increment numDocsWithWord
				for(j=0; j<uw_idx; j++) {
					if(!strcmp(unique_words[j].word, word)){
						contains = 1;
						if(unique_words[j].currDoc != i) {
							unique_words[j].numDocsWithWord++;
							unique_words[j].currDoc = i;
						}
						break;
					}
				}
			
				// If unique_words array does not contain it, make a new one with numDocsWithWord=1 
				if(!contains) {
					strcpy(unique_words[uw_idx].word, word);
					unique_words[uw_idx].numDocsWithWord = 1;
					unique_words[uw_idx].currDoc = i;
					uw_idx++;
				}
			}
			fclose(fp);
		}
	
	
		//each worker will send u_w sturct to the master
		//so that master can aggregate all the u_w word
		//structures into a single structure
		MPI_Send(&uw_idx, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
		MPI_Send(unique_words, sizeof(u_w) * uw_idx, MPI_BYTE, MASTER, 0, MPI_COMM_WORLD);

		//receive update unique words and numDocWithWords count and populate the current TFIDF obj struct array
		MPI_Recv(&uw_idx, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(unique_words, sizeof(u_w) * uw_idx, MPI_BYTE, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
		for(i=0; i<TF_idx; i++) {
			for(j=0; j<uw_idx; j++) {
				if(!strcmp(TFIDF[i].word, unique_words[j].word)) {
					TFIDF[i].numDocsWithWord = unique_words[j].numDocsWithWord;	
					break;
				}
			}
		}		
	
		// Calculates TFIDF value and store in TFIDF obj
		for(j=0; j<TF_idx; j++) {
			double TF = 1.0 * TFIDF[j].wordCount / TFIDF[j].docSize;
			double IDF = log(1.0 * TFIDF[j].numDocs / TFIDF[j].numDocsWithWord);
			double TFIDF_value = TF * IDF;
			TFIDF[j].val = TFIDF_value;
		}
		
		//send TFIDF values to the MASTER	
		MPI_Send(&TF_idx, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
		MPI_Send(TFIDF, sizeof(obj) * TF_idx, MPI_BYTE, MASTER, 0, MPI_COMM_WORLD);

	}

	MPI_Finalize();
	
	return 0;	
}
