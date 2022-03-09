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

    // 通过scatter分发数据，然后大家一起干活，然后gather汇总数据
    // 注意，分配offset等主进程专属工作应该在0号进程专属代码块里，但是scatter和gather不需要

    // 主进程计算任务
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
            // 每个进程分配到的任务数量
            task_num[i] =  num_of_reads / num_process;
            task_num[i] += i < num_of_reads % num_process ? 1 : 0;

            // 分发reads_CSR_offset需要的数据
            reads_offs_len[i] = task_num[i] + 1;
            reads_offs_off[i] = off_offset;
            off_offset += task_num[i];

            for (size_t k = 0; k < task_num[i]; j++, k++) {
                task_len[i] += reads_CSR_offs[j+1] - reads_CSR_offs[j];
            }   
            task_offset[i] = offset;
            offset += task_len[i];
            // printf("[%d] task_len = %d, task_num = %d, task_offset = %d\n", i, task_len[i], task_num[i], task_offset[i]);
        }
    }

    // scatter分配任务
    // 分发num_of_reads
    int local_reads_num;
    MPI_Scatter(task_num, 1, MPI_INT, &local_reads_num, 1, MPI_INT, 0, comm);
    
    // 分发reads_CSR_offset
    int *local_offs = new int[local_reads_num+5];
    MPI_Scatterv(reads_CSR_offs, reads_offs_len, reads_offs_off, MPI_INT, local_offs, local_reads_num + 1, MPI_INT, 0, comm);
    
    // 分发reads_CSR
    int local_len = local_offs[local_reads_num] - local_offs[0];
    char *local_reads_CSR = new char[local_len+5];
    MPI_Scatterv(reads_CSR, task_len, task_offset, MPI_CHAR, local_reads_CSR, local_len, MPI_CHAR, 0, comm);

    // 注意刚收到的offset，起点是按照原字符串算的。在每个进程里，起点都应该被重置为0
    // printf("final offset %d\n", local_offs[local_reads_num]);
    for (size_t i = 0, start = local_offs[0]; i <= local_reads_num; i++) {
        local_offs[i] -= start;
    }
    // printf("process %d has %d tasks, csv of length %d, range (%d, %d)\n", my_rank, local_reads_num, local_len, local_offs[0], local_offs[local_reads_num]);

    // 处理自己的任务
    vector<string> local_supermers;
    for (size_t i = 0; i < local_reads_num; i++) {
        read2supermers(
            local_reads_CSR + local_offs[i],
            local_offs[i+1] - local_offs[i],
            K, P,
            local_supermers
        );
    }
    // printf("this is process %d, range (%d, %d), %d results\n", my_rank, local_offs[0], local_offs[local_reads_num], local_supermers.size());

    

    int local_size = local_supermers.size();
    char* local_supermers_CSR;
    int *local_supermers_CSR_offs;
    vector<int> local_supermers_CSR_len(local_size);
    Vector2CSR(local_supermers, local_size, local_supermers_CSR, local_supermers_CSR_offs);


    local_supermers_CSR[local_supermers_CSR_offs[local_size]] = '\0';

    for (int i = 0; i < local_size; i++){
        local_supermers_CSR_len[i] = local_supermers[i].size();
    }

    // gather汇总任务
    int *gather_size;
    if(my_rank == 0){
        gather_size = new int[num_process];
        
    }
    // 每个进程返回答案数量
    MPI_Gather(&local_size, 1, MPI_INT, gather_size, 1, MPI_INT, 0, comm);
    
    int total_size = 0;
    int *CSR_len_displs;
    int *gather_CSR_len;
    if(my_rank == 0) {
        int offset = 0;

        CSR_len_displs = new int[num_process];
        for (size_t i = 0; i < num_process; i++) {
            printf("process %d has %d results\n", i, gather_size[i]);
            total_size += gather_size[i];
            CSR_len_displs[i] = offset;
            offset += gather_size[i];
        }

        printf("total results = %d\n", total_size);
        gather_CSR_len = new int[total_size];
    }

    // 尝试接收长度，而非offset。接收长度方便，收到后再处理
    // MPI_Gatherv(local_supermers_CSR_offs, local_size + 1, MPI_INT, gather_CSR_offs, gather_offs_num, gather_offs_off, MPI_INT, 0, comm);

    MPI_Gatherv(local_supermers_CSR_len.data(), local_size, MPI_INT, gather_CSR_len, gather_size, CSR_len_displs, MPI_INT, 0, comm);
    

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
    
    if(my_rank == 0){
        gather_CSR[total_CSR_len] = '\0';
        for (size_t i = 0, j = 0, p = 0; i < num_process; i++) {
            for (size_t k = 0; k < gather_size[i]; k++, j++) {
                all_supermers.emplace_back(gather_CSR + p, gather_CSR_len[j]);
                p += gather_CSR_len[j];
            }
        }
        delete []gather_size;
        delete []gather_CSR_len;
        delete []CSR_len_displs;
        delete []gather_CSR;
        delete []recvcount;
        delete []displs;
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
