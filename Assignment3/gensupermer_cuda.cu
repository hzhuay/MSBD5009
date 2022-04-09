#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <thread>
#include <future>
#include <string>
#include "gensupermer.hpp"

#include "utilities.hpp"
using namespace std;

__device__ __constant__ static const unsigned char d_basemap[256] = {
    255, 255, 255, 255, 255, 255, 255, 255, // 0..7
    255, 255, 255, 255, 255, 255, 255, 255, // 8..15
    255, 255, 255, 255, 255, 255, 255, 255, // 16..23
    255, 255, 255, 255, 255, 255, 255, 255, // 24..31
    255, 255, 255, 255, 255, 255, 255, 255, // 32..39
    255, 255, 255, 255, 255, 255, 255, 255, // 40..47
    255, 255, 255, 255, 255, 255, 255, 255, // 48..55
    255, 255, 255, 255, 255, 255, 255, 255, // 56..63
    255, 0, 255, 1, 255, 255, 255, 2, // 64..71
    255, 255, 255, 255, 255, 255, 255, 255, // 72..79
    255, 255, 255, 255, 3, 255, 255, 255, // 80..87
    255, 255, 255, 255, 255, 255, 255, 255, // 88..95
    255, 0, 255, 1, 255, 255, 255, 2, // 96..103
    255, 255, 255, 255, 255, 255, 255, 255, // 104..111
    255, 255, 255, 255, 3, 255, 255, 255, // 112..119
    255, 255, 255, 255, 255, 255, 255, 255, // 120..127
    255, 255, 255, 255, 255, 255, 255, 255, // 128..135
    255, 255, 255, 255, 255, 255, 255, 255, // 136..143
    255, 255, 255, 255, 255, 255, 255, 255, // 144..151
    255, 255, 255, 255, 255, 255, 255, 255, // 152..159
    255, 255, 255, 255, 255, 255, 255, 255, // 160..167
    255, 255, 255, 255, 255, 255, 255, 255, // 168..175
    255, 255, 255, 255, 255, 255, 255, 255, // 176..183
    255, 255, 255, 255, 255, 255, 255, 255, // 184..191
    255, 255, 255, 255, 255, 255, 255, 255, // 192..199
    255, 255, 255, 255, 255, 255, 255, 255, // 200..207
    255, 255, 255, 255, 255, 255, 255, 255, // 208..215
    255, 255, 255, 255, 255, 255, 255, 255, // 216..223
    255, 255, 255, 255, 255, 255, 255, 255, // 224..231
    255, 255, 255, 255, 255, 255, 255, 255, // 232..239
    255, 255, 255, 255, 255, 255, 255, 255, // 240..247
    255, 255, 255, 255, 255, 255, 255, 255  // 248..255
};

typedef struct {
    _in_ T_read_count cur_batch_size;    //number of reads
    
    // Raw reads
    _in_ _out_ char *reads;              //read_CSR 
    _in_ T_CSR_capacity *reads_offs;     //read_CSR_offset 
    _in_ _out_ T_read_len *read_len;      //length of each read
    
    // Minimizers
    _out_ _tmp_ T_minimizer *minimizers;  //the minimizer array
    _out_ T_read_len *supermer_offs;     //the supermer offset array
} T_GPU_data;

/*
 * [INPUT]  data.reads in [(Read#0), (Read#1)...]
 * [OUTPUT] data.minimizers in [(Read#0)[mm1, mm?, mm?, ...], (Read#1)...]
 */
__global__ void GPU_GenMinimizer(_in_ _out_ T_GPU_data data, int K_kmer, int P_minimizer) {

    // Fill in the GPU_GenMinimizer function here

    return;
}


/* [INPUT]  data.minimizers in [[mm1, mm1, mm2, mm3, ...], ...]
 * [OUTPUT] data.supermer_offs in [[0, 2, 3, ...], ...]
 */
__global__ void GPU_GenSKM(_in_ _out_ T_GPU_data data, int K_kmer, int P_minimizer) {

    // Fill in the GPU_GenSKM function here

    return;
}


void GenerateSupermer_GPU(vector<string> &reads, int K, int P, vector<string> &all_supermers, int NUM_BLOCKS_PER_GRID, int NUM_THREADS_PER_BLOCK) {


    T_GPU_data d_batch_data, h_batch_data;
    T_read_count cur_batch_size;

    CSR<char, int> csr_reads(true);
        
    // convert raw read from vector to csr:
    int name_idx_in_batch = 0;
    for(string read: reads) {
        csr_reads.append(read.c_str(), read.length(),name_idx_in_batch);
        name_idx_in_batch++;
    }
    cur_batch_size = reads.size();
    // malloc and memcpy from host to device
    h_batch_data.reads = new char[csr_reads.size()];
    h_batch_data.reads_offs = csr_reads.get_raw_offs();
    h_batch_data.read_len = new T_read_len[csr_reads.items()];
    h_batch_data.minimizers = new T_minimizer[csr_reads.size()];
    h_batch_data.supermer_offs = new T_read_len[csr_reads.size()];
    for (int i=0; i<name_idx_in_batch-1; i++) {
        h_batch_data.read_len[i] = csr_reads.get_raw_offs()[i+1] - csr_reads.get_raw_offs()[i];
    }
    h_batch_data.read_len[name_idx_in_batch-1] = csr_reads.size() - csr_reads.get_raw_offs()[name_idx_in_batch-1];

    // [Data H2D]
    // gpu malloc
    GPUErrChk(cudaMalloc((void**) &(d_batch_data.reads), sizeof(char) * csr_reads.size()));
    GPUErrChk(cudaMalloc((void**) &(d_batch_data.reads_offs), sizeof(size_t) * (csr_reads.items()+1)));
    d_batch_data.cur_batch_size = cur_batch_size;
    #ifdef DEBUG
    // cerr << "(GPU"<<GPU_ID<<"): batch_size = " << cur_batch_size << ", bases = " << csr_reads.size() << endl;
    // cerr << " <G" << GPU_ID << "> " << cur_batch_size << "|" << csr_reads.size();
    #endif
    GPUErrChk(cudaMalloc((void**) &(d_batch_data.read_len), sizeof(T_read_len) * csr_reads.items()));
    GPUErrChk(cudaMalloc((void**) &(d_batch_data.minimizers), sizeof(T_minimizer) * csr_reads.size()));
    GPUErrChk(cudaMalloc((void**) &(d_batch_data.supermer_offs), sizeof(T_read_len) * csr_reads.size()));
    // memcpy Host -> Device
    GPUErrChk(cudaMemcpy(d_batch_data.reads, csr_reads.get_raw_data(), sizeof(char) * csr_reads.size(), cudaMemcpyHostToDevice));
    GPUErrChk(cudaMemcpy(d_batch_data.read_len, h_batch_data.read_len, sizeof(T_read_len) * (csr_reads.items()), cudaMemcpyHostToDevice));
    GPUErrChk(cudaMemcpy(d_batch_data.reads_offs, csr_reads.get_raw_offs(), sizeof(size_t) * (csr_reads.items()+1), cudaMemcpyHostToDevice));

    // [Computing]
    GPU_GenMinimizer<<<NUM_BLOCKS_PER_GRID, NUM_THREADS_PER_BLOCK/*, 0, cuda_stream*/>>>(d_batch_data, K, P);
    GPU_GenSKM<<<NUM_BLOCKS_PER_GRID, NUM_THREADS_PER_BLOCK/*, 0, cuda_stream*/>>>(d_batch_data, K, P);

    cudaDeviceSynchronize();

    // [Data D2H]
    
    GPUErrChk(cudaMemcpy(h_batch_data.reads, d_batch_data.reads, sizeof(char) * (csr_reads.size()), cudaMemcpyDeviceToHost));
    GPUErrChk(cudaMemcpy(h_batch_data.read_len, d_batch_data.read_len, sizeof(T_read_len) * (csr_reads.items()), cudaMemcpyDeviceToHost));
    GPUErrChk(cudaMemcpy(h_batch_data.minimizers, d_batch_data.minimizers, sizeof(T_minimizer) * (csr_reads.size()), cudaMemcpyDeviceToHost));
    GPUErrChk(cudaMemcpy(h_batch_data.supermer_offs, d_batch_data.supermer_offs, sizeof(T_read_len) * (csr_reads.size()), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    // GPUErrChk(cudaStreamSynchronize(cuda_stream));
    GPUErrChk(cudaFree(d_batch_data.reads));
    GPUErrChk(cudaFree(d_batch_data.reads_offs));
    GPUErrChk(cudaFree(d_batch_data.minimizers));
    GPUErrChk(cudaFree(d_batch_data.read_len));
    GPUErrChk(cudaFree(d_batch_data.supermer_offs));
    cudaDeviceSynchronize();
    
    for (int i=0; i<name_idx_in_batch; i++) {
        int skm_idx = 1;
        T_read_len *skm_offs = &h_batch_data.supermer_offs[h_batch_data.reads_offs[i]];
        vector<string>  supermers_local_all;
        T_read_len len = h_batch_data.read_len[i];
        while (*(skm_offs+skm_idx-1) != len- K +1) {
            int skm_len = *(skm_offs+skm_idx) - *(skm_offs+skm_idx-1) + K-1;
            char* t = (char *) malloc(skm_len+1);
            memcpy(t, h_batch_data.reads +h_batch_data.reads_offs[i]+ *(skm_offs+skm_idx-1), skm_len);
            t[skm_len] = '\0';
            supermers_local_all.push_back(t);
            skm_idx++;
        }

        all_supermers.insert(all_supermers.end(), supermers_local_all.begin(), supermers_local_all.end());
    }
    

    delete [] h_batch_data.reads;
    delete [] h_batch_data.read_len;
    delete [] h_batch_data.minimizers;
    delete [] h_batch_data.supermer_offs;


    return;
}
