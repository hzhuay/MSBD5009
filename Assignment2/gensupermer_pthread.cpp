/**********************************************************************
    [NAME]:    
    [STUDENT ID]: 
    [EMAIL]: 
    NOTICE: Write your own code only in this file.

    COMPILE: g++ -std=c++11 -lpthread gensupermer_pthread.cpp main.cpp -o pthread_gs
                      ps, in some platform, the command "-lpthread" should be replaced by "-pthread"
    RUN:     ./pthread_gs <K> <P> <num_threads> <data_file> <correctness_check> <result_folder (optional)>
**********************************************************************/



#define _in_
#define _out_


#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <cmath>
#include <pthread.h>
#include <semaphore.h>
#include <vector>
#include "gensupermer.hpp"

using namespace std;

vector<string> supermers_tol;

void read2supermers(const char* _read, int read_len, int k, int p, _out_ vector<string> &supermers);

struct AllThings
{
    /*
        AllThings contains all arguments that are passed to the thread function.
    */
    /*========== Add elements and construct AllThings below this line ==========*/
    


    /*==========Add elements and construct AllThings above this line    ==========*/
};

void *parallel(void *allthings)
{
    /*
        Thread function - the function that threads are to run.
        You need to implement the thread function parallel to generate supermers.
        You can call the function read2supermers which has been implemented.
    */
    /*========== Fill the body of your thread function below this line ==========*/
    
    /*========== Fill the body of your thread function above this line ==========*/

    return 0;

}

int gensupermers(char *reads, int *reads_offs, int K, int P, int num_of_reads, vector<string> &all_supermers, int num_threads){
    /*
        Main function will call this gensupermers to generate supermers from reads.
        In this function, you need to initiate threads, call the thread function and
        store the results into the vector of strings all_supermers.

        Output: all_supermers - you need to save your results in the vector all_supermers
    */
    /*========== Fill in your code below this line ==========*/

    /*========== Fill in your code above this line ==========*/

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