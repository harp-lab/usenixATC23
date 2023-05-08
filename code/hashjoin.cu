#include <cstdio>
#include <string>
#include <iostream>
#include <chrono>
#include <math.h>
#include <iomanip>
#include <assert.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/set_operations.h>
#include "common/error_handler.cu"
#include "common/utils.cu"
#include "common/kernels.cu"


using namespace std;

void gpu_hashjoin(const char *data_path, char separator,
                  long int relation_rows, double load_factor,
                  int preferred_grid_size, int preferred_block_size, const char *dataset_name,
                  int number_of_sm, int random_category) {
    KernelTimer timer;
    int relation_columns = 2;
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    time_point_begin = chrono::high_resolution_clock::now();
    double spent_time;
    output.initialization_time = 0;
    output.join_time = 0;
    output.memory_clear_time = 0;
    output.total_time = 0;

    int block_size, grid_size;
    int *relation;
    int *relation_host;
    Entity *hash_table;
    Entity *result_host;
    long int join_result_rows;
    long int hash_table_rows = (long int) relation_rows / load_factor;
    hash_table_rows = pow(2, ceil(log(hash_table_rows) / log(2)));

    checkCuda(cudaMallocHost((void **) &relation_host, relation_rows * relation_columns * sizeof(int)));
    checkCuda(cudaMalloc((void **) &relation, relation_rows * relation_columns * sizeof(int)));
    checkCuda(cudaMalloc((void **) &hash_table, hash_table_rows * sizeof(Entity)));
    block_size = 512;
    grid_size = 32 * number_of_sm;
    if (preferred_grid_size != 0) {
        grid_size = preferred_grid_size;
    }
    if (preferred_block_size != 0) {
        block_size = preferred_block_size;
    }
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.initialization_time += spent_time;
    if (random_category == -1) {
        get_relation_from_file_gpu(relation_host, data_path, relation_rows, relation_columns, separator);
    } else if (random_category == 1) {
        get_random_relation(relation_host, relation_rows, relation_columns);
    } else if (random_category == 2) {
        get_string_relation(relation_host, relation_rows, relation_columns);
    }
//    show_relation(relation_host, relation_rows, 2, "relation", 10, -1);
    time_point_begin = chrono::high_resolution_clock::now();
    cudaMemcpy(relation, relation_host, relation_rows * relation_columns * sizeof(int),
               cudaMemcpyHostToDevice);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.read_time = spent_time;

    Entity negative_entity;
    negative_entity.key = -1;
    negative_entity.value = -1;
    time_point_begin = chrono::high_resolution_clock::now();
    thrust::fill(thrust::device, hash_table, hash_table + hash_table_rows, negative_entity);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.initialization_time += spent_time;

    timer.start_timer();
    build_hash_table<<<grid_size, block_size>>>
            (hash_table, hash_table_rows,
             relation, relation_rows,
             relation_columns);
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    double kernel_spent_time = timer.get_spent_time();
    double rate = (double) relation_rows / kernel_spent_time;
    output.hashtable_build_time = kernel_spent_time;
    double build_rate = (double) relation_rows / kernel_spent_time;
    output.hashtable_build_rate = build_rate;
    output.join_time += kernel_spent_time;

    time_point_begin = chrono::high_resolution_clock::now();
    int *offset;
    Entity *join_result;
    checkCuda(cudaMalloc((void **) &offset, relation_rows * sizeof(int)));
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.join_time += spent_time;
    timer.start_timer();
    get_join_result_size_ar<<<grid_size, block_size>>>(hash_table, hash_table_rows, relation, relation_rows, offset);
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    output.join_time += timer.get_spent_time();
    time_point_begin = chrono::high_resolution_clock::now();
    join_result_rows = thrust::reduce(thrust::device, offset, offset + relation_rows, 0);
    thrust::exclusive_scan(thrust::device, offset, offset + relation_rows, offset);
    checkCuda(cudaMalloc((void **) &join_result, join_result_rows * sizeof(Entity)));
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.join_time += spent_time;
    timer.start_timer();
    get_join_result_ar<<<grid_size, block_size>>>(hash_table, hash_table_rows, relation, relation_rows,
                                                  offset, join_result);
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    output.join_time += timer.get_spent_time();
    time_point_begin = chrono::high_resolution_clock::now();
    checkCuda(cudaMallocHost((void **) &result_host, join_result_rows * sizeof(Entity)));
    cudaMemcpy(result_host, join_result, join_result_rows * sizeof(Entity),
               cudaMemcpyDeviceToHost);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.join_time += spent_time;
//    show_entity_array(result_host, join_result_rows, "Join Result");
    time_point_begin = chrono::high_resolution_clock::now();
    cudaFree(hash_table);
    cudaFreeHost(result_host);
    cudaFree(relation);
    cudaFreeHost(relation_host);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.memory_clear_time += spent_time;
    double calculated_time = output.initialization_time +
                             output.read_time + output.hashtable_build_time + output.join_time +
                             output.memory_clear_time;
    cout << endl;
    cout << "| Dataset | Number of rows | #Join | Blocks x Threads | Time (s) |" << endl;
    cout << "| --- | --- | --- | --- | --- |" << endl;
    cout << "| " << dataset_name << " | " << relation_rows << " | " << join_result_rows << " | ";
    cout << fixed << grid_size << " x " << block_size << " | " << calculated_time << " |\n" << endl;
    output.block_size = block_size;
    output.grid_size = grid_size;
    output.input_rows = relation_rows;
    output.load_factor = load_factor;
    output.hashtable_rows = hash_table_rows;
    output.dataset_name = dataset_name;
    output.total_time = calculated_time;

    cout << "Initialization: " << output.initialization_time;
    cout << ", Read: " << output.read_time << endl;
    cout << "Hashtable rate: " << output.hashtable_build_rate << " keys/s, time: ";
    cout << output.hashtable_build_time << endl;
    cout << "Join: " << output.join_time << endl;
    cout << "Memory clear: " << output.memory_clear_time << endl;
    cout << "Total: " << output.total_time << endl;
}


void run_benchmark(int grid_size, int block_size, double load_factor, int random) {
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, device_id);
    std::locale loc("");
    std::cout.imbue(loc);
    char separator = '\t';
    string datasets[] = {
            "OL.cedge_initial", "../data/data_7035.txt",
            "CA-HepTh", "../data/data_51971.txt",
            "SF.cedge", "../data/data_223001.txt",
            "ego-Facebook", "../data/data_88234.txt",
            "wiki-Vote", "../data/data_103689.txt",
            "p2p-Gnutella09", "../data/data_26013.txt",
            "p2p-Gnutella04", "../data/data_39994.txt",
            "cal.cedge", "../data/data_21693.txt",
            "TG.cedge", "../data/data_23874.txt",
            "OL.cedge", "../data/data_7035.txt",
            "luxembourg_osm", "../data/data_119666.txt",
            "fe_sphere", "../data/data_49152.txt",
            "fe_body", "../data/data_163734.txt",
            "cti", "../data/data_48232.txt",
            "fe_ocean", "../data/data_409593.txt",
            "wing", "../data/data_121544.txt",
            "loc-Brightkite", "../data/data_214078.txt",
            "delaunay_n16", "../data/data_196575.txt",
            "usroads", "../data/data_165435.txt",
//            "usroads-48", "../data/data_161950.txt",
//            "String 9990", "../data/data_9990.txt",
//            "String 2990", "../data/data_2990.txt",
//            "talk 5", "../data/data_5.txt",
//            "string 4", "../data/data_4.txt",
//            "cyclic 3", "../data/data_3.txt",
    };

    string random_datasets[] = {
//            "random 1000",
//            "random 2000",
//            "string 4000",
//            "string 5000",
            "random 1000000",
            "random 2000000",
            "random 3000000",
            "random 4000000",
            "random 5000000",
            "string 1000000",
            "string 2000000",
            "string 3000000",
            "string 4000000",
            "string 5000000",
    };


    if (random == 1) {
        for (int i = 0; i < sizeof(random_datasets) / sizeof(random_datasets[0]); i++) {
            int random_category = 1;
            if (random_datasets[i].compare(0, 6, "string") == 0)
                random_category = 2;
            const char *dataset_name;
            dataset_name = random_datasets[i].c_str();
            long int row_size = get_row_size(dataset_name);
            cout << "Benchmark for " << dataset_name << endl;
            cout << "----------------------------------------------------------" << endl;
            gpu_hashjoin("", separator,
                         row_size, load_factor,
                         grid_size, block_size, dataset_name, number_of_sm, random_category);
            cout << endl;
        }
    } else {
        for (int i = 0; i < sizeof(datasets) / sizeof(datasets[0]); i += 2) {
            const char *data_path, *dataset_name;
            dataset_name = datasets[i].c_str();
            data_path = datasets[i + 1].c_str();
            long int row_size = get_row_size(data_path);
            cout << "Benchmark for " << dataset_name << endl;
            cout << "----------------------------------------------------------" << endl;
            gpu_hashjoin(data_path, separator,
                         row_size, load_factor,
                         grid_size, block_size, dataset_name, number_of_sm, -1);
            cout << endl;

        }
    }

}

int main() {
    // set the last parameter to 1 for random graphs
    run_benchmark(0, 0, 0.1, 1);
    return 0;
}

// Benchmark
// nvcc hashjoin.cu -run -o hashjoin.out
// or
// make hashjoin
