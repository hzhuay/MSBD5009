// *********************************************************************
//     [NAME]: ZHU Haunqi   [STUDENT ID]: 20786002
//     [EMAIL]: hzhuay@connect.ust.hk
//     NOTICE: Write your code only in the specified section.
// *********************************************************************
// 7 MAR (update2.1), 28 FEB (update1): UPDATES IN read2supermers(...)
#define _in_
#define _out_
#define _MPI_TEST_
// #define DEBUG
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include "utilities.hpp"
#ifdef _MPI_TEST_
#include "mpi.h"
#endif
using namespace std;

// void read2supermers(const char* _read, int read_len, int k, int p, _out_ char* &supermers, _out_ /**/int* &supermer_offs, int &n_supermers);
void read2supermers(const char* _read, int read_len, int k, int p, _out_ vector<string> &supermers);

const int MAX_PROCESS = 64;
int K, P;

int main(int argc, char **argvs) {
    #ifdef _MPI_TEST_
    MPI_Init(&argc, &argvs);
    MPI_Comm comm;
    int num_process; // number of processors
    int my_rank;     // my global rank
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_process);
    MPI_Comm_rank(comm, &my_rank);
    #endif

    int correctness_checking = 0;
    string output_path;
    string read_file_path;
    ArgParser(argc, argvs, K, P, read_file_path, correctness_checking, output_path);
    vector<string> reads;
    
    // Input data (the reads in CSR format)
    int num_of_reads = 0;
    char* reads_CSR;
    /**/int* reads_CSR_offs;

    // Output data, each supermers should be a string in the vector
    // you need to save all the supermers to the vector below in the root(0) process
    vector<string> all_supermers;

    #ifdef _MPI_TEST_
    if (my_rank == 0) {
        cout<<"MPI running with "<<num_process<<" threads."<<endl<<endl;
    #endif
        LoadReadsFromFile(read_file_path.c_str(), reads);
        Vector2CSR(reads, num_of_reads, reads_CSR, reads_CSR_offs);
        cout << reads.size() << " reads loaded from "<< read_file_path << endl << endl;
    #ifdef _MPI_TEST_
    }
    #endif

    // time measurement starts
    auto start_time = chrono::high_resolution_clock::now();
    
    #ifdef _MPI_TEST_
    // hint: Input: "num_of_reads", "reads_CSR", "reads_CSR_offs", "K", "P"
    //       You need to save all the generated supermers in the vector "all_supermers" in Process 0.
    // you need to do:
    //       1. Scatter the read data to each MPI processes.
    //       2. Perform the super-mer generation in each process. 
    //          (you can refer to the sequential version to know the usage of the function read2supermers(...))
    //       3. Gather all the super-mers to the root process and store in the vector "all_supermers". (The order in the vector doesn't matter.)
    
    // ==============================================================
    // ==============================================================
    // ====    Write your implementation only below this line    ====
    // ==============================================================

    // 1. Scatter the read data to each MPI processes.
    int *task_len = nullptr, *task_num = nullptr, *task_offset = nullptr;
    int *reads_offs_off = nullptr, *reads_offs_len = nullptr;
    if (my_rank == 0) {
        task_len = new int[num_process]();
        task_num = new int[num_process];
        task_offset = new int[num_process];
        reads_offs_off = new int[num_process + 1];
        reads_offs_len = new int[num_process];

        int j = 0, offset = 0, off_offset = 0;
        for (size_t i = 0; i < num_process; i++) {
            // how many reads are assigned to task[i]
            task_num[i] =  num_of_reads / num_process;
            task_num[i] += i < num_of_reads % num_process ? 1 : 0;

            // scatter reads_CSR_offset
            reads_offs_len[i] = task_num[i] + 1;
            reads_offs_off[i] = off_offset;
            off_offset += task_num[i];

            for (size_t k = 0; k < task_num[i]; j++, k++) {
                task_len[i] += reads_CSR_offs[j+1] - reads_CSR_offs[j];
            }   
            task_offset[i] = offset;
            offset += task_len[i];
        }
    }

    // scatter the data to each process
    // scatter num_of_reads
    int local_reads_num;
    MPI_Scatter(task_num, 1, MPI_INT, &local_reads_num, 1, MPI_INT, 0, comm);
    
    // scatter reads_CSR_offset
    int *local_offs = new int[local_reads_num+5];
    MPI_Scatterv(reads_CSR_offs, reads_offs_len, reads_offs_off, MPI_INT, local_offs, local_reads_num + 1, MPI_INT, 0, comm);
    
    // scatter reads_CSR
    int local_len = local_offs[local_reads_num] - local_offs[0];
    char *local_reads_CSR = new char[local_len+5];
    MPI_Scatterv(reads_CSR, task_len, task_offset, MPI_CHAR, local_reads_CSR, local_len, MPI_CHAR, 0, comm);

    // set all offests to start at 0 for each processes
    for (size_t i = 0, start = local_offs[0]; i <= local_reads_num; i++) {
        local_offs[i] -= start;
    }

    if (my_rank == 0) {
        delete [] task_len;
        delete [] task_num;
        delete [] task_offset;
        delete [] reads_offs_off;
        delete [] reads_offs_len;
    }

    // 2. Perform the super-mer generation in each process. 
    vector<string> local_supermers;
    for (size_t i = 0; i < local_reads_num; i++) {
        read2supermers(
            local_reads_CSR + local_offs[i],
            local_offs[i+1] - local_offs[i],
            K, P,
            local_supermers
        );
    }

    int local_size = local_supermers.size();
    char* local_supermers_CSR;
    int *local_supermers_CSR_offs;
    Vector2CSR(local_supermers, local_size, local_supermers_CSR, local_supermers_CSR_offs);

    // append an end-of-string character
    local_supermers_CSR[local_supermers_CSR_offs[local_size]] = '\0';

    // create a vector to store the length of each local_supermers_CSR
    vector<int> local_supermers_CSR_len(local_size);
    for (int i = 0; i < local_size; i++){
        local_supermers_CSR_len[i] = local_supermers[i].size();
    }

    delete [] local_offs;
    delete [] local_reads_CSR;

    // 3. Gather all the super-mers to the root process
    
    // gather the number of supermers each process returns
    int *gather_size;
    if(my_rank == 0){
        gather_size = new int[num_process];
    }
    MPI_Gather(&local_size, 1, MPI_INT, gather_size, 1, MPI_INT, 0, comm);
    
    // gather the length of every supermers from each process
    int total_size = 0;
    int *CSR_len_displs;
    int *gather_CSR_len;
    if(my_rank == 0) {
        int offset = 0;
        CSR_len_displs = new int[num_process];
        for (size_t i = 0; i < num_process; i++) {
            // printf("process %d has %d results\n", i, gather_size[i]);
            total_size += gather_size[i];
            CSR_len_displs[i] = offset;
            offset += gather_size[i];
        }

        // printf("total results = %d\n", total_size);
        gather_CSR_len = new int[total_size];
    }
    MPI_Gatherv(local_supermers_CSR_len.data(), local_size, MPI_INT, gather_CSR_len, gather_size, CSR_len_displs, MPI_INT, 0, comm);
    

    // gather the supermers_CSR from each process
    int total_CSR_len = 0;
    char* gather_CSR;
    int *recvcount, *displs;
    if(my_rank == 0){
        recvcount = new int[num_process]();
        displs = new int[num_process]();
        int offset = 0;
        for (size_t i = 0, j = 0; i < num_process; i++) {
            for (size_t k = 0; k < gather_size[i]; k++, j++) {
                // printf("process %d has %d results, result[%d] len = %d\n", i, gather_size[i], j, gather_CSR_len[j]);
                total_CSR_len += gather_CSR_len[j];
                recvcount[i] += gather_CSR_len[j];
            }
            displs[i] += offset;
            offset += recvcount[i];
        }
        gather_CSR = new char[total_CSR_len + 10];
    }

    int len = local_supermers_CSR_offs[local_size] - local_supermers_CSR_offs[0];
    MPI_Gatherv(local_supermers_CSR, len, MPI_CHAR, gather_CSR, recvcount, displs, MPI_CHAR, 0, comm); 
    
    // 4. store all supermers in the vector "all_supermers" and clean the resources
    if(my_rank == 0){
        gather_CSR[total_CSR_len] = '\0';
        for (size_t i = 0, j = 0, p = 0; i < num_process; i++) {
            for (size_t k = 0; k < gather_size[i]; k++, j++) {
                all_supermers.emplace_back(gather_CSR + p, gather_CSR_len[j]);
                p += gather_CSR_len[j];
            }
        }
        delete [] gather_size;
        delete [] gather_CSR_len;
        delete [] CSR_len_displs;
        delete [] gather_CSR;
        delete [] recvcount;
        delete [] displs;
    }
    delete [] local_supermers_CSR;
    delete [] local_supermers_CSR_offs;
    
    // ==============================================================
    // ====    Write your implementation only above this line    ====
    // ==============================================================
    // ==============================================================
    #endif
    
    #ifdef _MPI_TEST_
    if (my_rank == 0) {
    #endif
        // time measurement ends
        auto end_time = chrono::high_resolution_clock::now();
        auto duration_sec = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count()/1000.0;
        cout << "Your algorithm finished in " << duration_sec << " sec." << endl << endl;
        
        // output to text file and correctness checking
        delete reads_CSR;
        delete reads_CSR_offs;
        if (correctness_checking) CorrectnessChecking(reads, K, P, all_supermers);
        if (!output_path.empty()) {
            if (!correctness_checking) sort(all_supermers.begin(), all_supermers.end());
            SaveSupermers(output_path, all_supermers);
        }
    #ifdef _MPI_TEST_
    }
    MPI_Barrier(comm);
    // cout<<"Thread "<<my_rank<<" ends."<<endl;
    
    MPI_Finalize();
    #endif
    
    return 0;
}

/*
This function receives a C-style read string, the length of the read,
k (length of k-mer), p (length of miniizer), and output the supermers 
which can be generated from this read to a vector<string>.
*/
void read2supermers(const char* _read, int read_len, int k, int p, _out_ vector<string> &supermers) {
    string prev_minimizer, minimizer, new_minimizer;
    string read(_read, read_len); // from-buffer init
    int i, j;
    char base;
    int skm_begin_pos, skm_end_pos, mm_begin_pos;
    
    // Generate the first k-mer's minimizer:
    skm_begin_pos = 0;
    skm_end_pos = k;
    mm_begin_pos = 0;
    minimizer = new_minimizer = read.substr(0, p);
    for (i=p; i<k; i++) {
        new_minimizer = new_minimizer.substr(1, p-1) + read[i]; // UPDATE1
        if (new_minimizer <= minimizer) minimizer = new_minimizer, mm_begin_pos = i-p+1;
    }

    // Continue generating minimizers:
    for (i=1; i<read_len-k+1; i++) { // i: the beginning position of the current k-mer
        if (i > mm_begin_pos) {
            // new minimizer required
            prev_minimizer = minimizer;
            minimizer = new_minimizer = read.substr(i, p);
            for (j=i+p; j<i+k; j++) {
                new_minimizer = new_minimizer.substr(1, p-1) + read[j]; // UPDATE1
                if (new_minimizer <= minimizer) minimizer = new_minimizer, mm_begin_pos = j-p+1;
            }
            // if the new minimizer equals to the previous one, we can continue
            if (minimizer != prev_minimizer) {
                skm_end_pos = i-1+k;
                supermers.push_back(read.substr(skm_begin_pos, skm_end_pos-skm_begin_pos)); // save the supermer
                skm_begin_pos = i;
            }
        }
        else {
            new_minimizer = read.substr(i+k-p, p); // UPDATE1
            if (new_minimizer < minimizer) { // save the supermer
                skm_end_pos = i-1+k;
                supermers.push_back(read.substr(skm_begin_pos, skm_end_pos-skm_begin_pos));
                skm_begin_pos = i;
                minimizer = new_minimizer, mm_begin_pos = i+k-1-p+1;
            }
            if (new_minimizer == minimizer) mm_begin_pos = i+k-1-p+1; // UPDATE1
        }
    } // UPDATE 2.1
    skm_end_pos = read_len;
    supermers.push_back(read.substr(skm_begin_pos, skm_end_pos-skm_begin_pos));
}
