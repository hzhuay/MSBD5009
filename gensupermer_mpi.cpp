// *********************************************************************
//     [NAME]:    [STUDENT ID]: 
//     [EMAIL]: 
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

        // 主进程把任务分发完后，和子进程一起干活，然后汇总数据
    // 只有主进程读取了数据，因此肯定需要主进程给子进程分发任务，要把数据传递过去

    if (my_rank == 0) {
        int local_num = num_of_reads / num_process;
        // cout << strlen(reads_CSR) << endl;
        // 分配任务
        // printf("send %s\n", reads_CSR);
        for (size_t i = 0; i < num_process; i++) {
            // 给子进程发送任务
            if (i != 0) {
                MPI_Send(&num_of_reads, 1, MPI_INT, i, 0, comm);
                MPI_Send(reads_CSR_offs, num_of_reads + 1, MPI_INT, i, 0, comm);
                MPI_Send(reads_CSR, reads_CSR_offs[num_of_reads], MPI_CHAR, i, 0, comm);
            }
        }
        
        // 完成自己的任务
        
        for (int i = my_rank; i < num_of_reads; i += num_process) {
            read2supermers(
                reads_CSR + reads_CSR_offs[i],
                reads_CSR_offs[i + 1] - reads_CSR_offs[i],
                K, P,
                all_supermers
            );
            // printf("this is father %d doing task %d, range (%d, %d) %d\n", my_rank, i, reads_CSR_offs[i], reads_CSR_offs[i+1], all_supermers.size());
        }


        // 收取任务
        // puts("收任务");
        int* local_offs;
        char* local_reads_CSR;
        
        for (size_t i = 1; i < num_process; i++) {
            
            int size;
            MPI_Recv(&size, 1, MPI_INT, i, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
            
            local_offs = (int*)calloc(size, sizeof(int));
            MPI_Recv(local_offs, size + 1, MPI_INT, i, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);

            int supermer_CSR_size = local_offs[size];
            // printf("主进程从%d进程收取任务，数组长度为%d\n，字符串长度为%d\n", i, size, supermer_CSR_size);

            local_reads_CSR = (char*)calloc(supermer_CSR_size, sizeof(char));
            MPI_Recv(local_reads_CSR, supermer_CSR_size, MPI_CHAR, i, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);

            vector<string> supermers_local(size);
            for (size_t i = 0; i < size; i++) {
                all_supermers.emplace_back(local_reads_CSR + local_offs[i], local_offs[i + 1] - local_offs[i]);
            }

        }
        // puts("排序");
        sort(all_supermers.begin(), all_supermers.end());
        // 释放各种资源
        free(local_offs);
        free(local_reads_CSR);

    } else {
        
        MPI_Recv(&num_of_reads, 1, MPI_INT, 0, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
        // printf("%d %d\n", my_rank, num_of_reads);

        int* local_offs = (int*)calloc(num_of_reads + 1, sizeof(int));
        MPI_Recv(local_offs, num_of_reads + 1, MPI_INT, 0, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
        
        int reads_CSR_size = local_offs[num_of_reads];
        // printf("process %d, csr_size = %d\n", my_rank, reads_CSR_size);

        char* local_reads_CSR = (char*)calloc(reads_CSR_size, sizeof(char));
        MPI_Recv(local_reads_CSR, reads_CSR_size, MPI_CHAR, 0, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
    
        
        vector<string> supermers_local;

        for (int i = my_rank; i < num_of_reads; i += num_process) {
            // cout<<string(local_reads_CSR + local_offs[i], local_offs[i + 1] - local_offs[i])<<endl;
            // printf("this is process %d, task %d and string is %s, strlen is %d\n", my_rank, i, string(local_reads_CSR + local_offs[i], local_offs[i+1] - local_offs[i]).c_str(), local_offs[i+1] - local_offs[i]);
            read2supermers(
                local_reads_CSR + local_offs[i],
                local_offs[i + 1] - local_offs[i],
                K, P,
                supermers_local
            );
            // printf("this is child %d doing task %d, range (%d, %d), start at %c, result size %d %s\n", my_rank, i, local_offs[i], local_offs[i+1], local_reads_CSR[local_offs[i]], supermers_local.size(), supermers_local[0].c_str());
        }
        //vector<string>无法直接发送，因此要转换成CSR
        int size = supermers_local.size();
        Vector2CSR(supermers_local, size, reads_CSR, reads_CSR_offs);
        
        MPI_Send(&size, 1, MPI_INT, 0, 0, comm);

        MPI_Send(reads_CSR_offs, size + 1, MPI_INT, 0, 0, comm);

        MPI_Send(reads_CSR, reads_CSR_offs[size], MPI_CHAR, 0, 0, comm);
        
        free(local_offs);
        free(local_reads_CSR);
       
    }
    
    
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
