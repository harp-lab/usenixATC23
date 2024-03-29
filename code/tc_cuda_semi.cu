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
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/set_operations.h>
#include "common/error_handler.cu"
#include "common/utils.cu"
#include "common/kernels.cu"


using namespace std;

void gpu_tc(const char *data_path, char separator,
            long int relation_rows, double load_factor,
            int preferred_grid_size, int preferred_block_size, const char *dataset_name, int number_of_sm) {
    int relation_columns = 2;
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    std::chrono::high_resolution_clock::time_point temp_time_begin;
    std::chrono::high_resolution_clock::time_point temp_time_end;
    KernelTimer timer;
    time_point_begin = chrono::high_resolution_clock::now();
    double spent_time;
    output.initialization_time = 0;
    output.join_time = 0;
    output.projection_time = 0;
    output.deduplication_time = 0;
    output.memory_clear_time = 0;
    output.union_time = 0;
    output.total_time = 0;
    double sort_time = 0.0;
    double unique_time = 0.0;
    double merge_time = 0.0;
    double temp_spent_time = 0.0;

    int block_size, grid_size;
    Entity *relation;
    Entity *relation_host;
    Entity *hash_table, *t_full, *t_delta;
    Entity *t_full_host;

    checkCuda(cudaMallocHost((void **) &relation_host, relation_rows * sizeof(Entity)));
    checkCuda(cudaMalloc((void **) &relation, relation_rows * sizeof(Entity)));
    // Block size is 512 if preferred_block_size is 0
    block_size = 512;
    // Grid size is 32 times of the number of streaming multiprocessors if preferred_grid_size is 0
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
    time_point_begin = chrono::high_resolution_clock::now();
    get_relation_from_file_gpu_entity(relation_host, data_path,
                                      relation_rows, relation_columns, separator);
    cudaMemcpy(relation, relation_host, relation_rows * sizeof(Entity),
               cudaMemcpyHostToDevice);

    thrust::stable_sort(thrust::device, relation, relation + relation_rows, set_cmp());
    relation_rows = (thrust::unique(thrust::device,
                                    relation, relation + relation_rows,
                                    is_equal())) - relation;


    long int t_delta_rows = relation_rows;
    long int t_full_rows = relation_rows;
    long int iterations = 0;
    long int hash_table_rows = (long int) relation_rows / load_factor;
    hash_table_rows = pow(2, ceil(log(hash_table_rows) / log(2)));

    checkCuda(cudaMalloc((void **) &t_full, t_full_rows * sizeof(Entity)));
    checkCuda(cudaMalloc((void **) &t_delta, t_delta_rows * sizeof(Entity)));
    checkCuda(cudaMalloc((void **) &hash_table, hash_table_rows * sizeof(Entity)));


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
    build_hash_table_entity<<<grid_size, block_size>>>
            (hash_table, hash_table_rows,
             relation, relation_rows,
             relation_columns);
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    spent_time = timer.get_spent_time();
    output.hashtable_build_time = spent_time;
    output.hashtable_build_rate = (double) relation_rows / spent_time;
    output.join_time += spent_time;

    timer.start_timer();
    // initial result and t delta both are same as the input relation
    initialize_result_t_delta_entity<<<grid_size, block_size>>>(t_full,
                                                                t_delta, relation, relation_rows, relation_columns);
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    spent_time = timer.get_spent_time();
    output.union_time += spent_time;

    time_point_begin = chrono::high_resolution_clock::now();
    cudaFree(relation);
    cudaFreeHost(relation_host);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.memory_clear_time += spent_time;

    // Run the fixed point iterations for transitive closure computation
    while (true) {
        int *offset;
        long int join_result_rows;
        Entity *join_result;
        checkCuda(cudaMalloc((void **) &offset, t_delta_rows * sizeof(int)));
        // First pass to get the join result size for each row of t_delta
        get_join_result_size<<<grid_size, block_size>>>(hash_table, hash_table_rows,
                                                        t_delta, t_delta_rows, offset);
        checkCuda(cudaDeviceSynchronize());
        join_result_rows = thrust::reduce(thrust::device, offset, offset + t_delta_rows, 0);
        thrust::exclusive_scan(thrust::device, offset, offset + t_delta_rows, offset);
        checkCuda(cudaMalloc((void **) &join_result, join_result_rows * sizeof(Entity)));
        // Second pass to generate the join result of t_delta and the hash_table
        get_join_result<<<grid_size, block_size>>>(hash_table, hash_table_rows,
                                                   t_delta, t_delta_rows, offset, join_result);
        checkCuda(cudaDeviceSynchronize());
        thrust::stable_sort(thrust::device, join_result, join_result + join_result_rows, set_cmp());
        long int unique_join_result_rows = thrust::unique(thrust::device,
                                                          join_result, join_result + join_result_rows,
                                                          is_equal()) - join_result;

        Entity *new_t_full;
        long int new_t_full_rows = unique_join_result_rows + t_full_rows;
        checkCuda(cudaMalloc((void **) &new_t_full, new_t_full_rows * sizeof(Entity)));
        new_t_full_rows = thrust::set_union(thrust::device,
                                            t_full, t_full + t_full_rows,
                                            join_result, join_result + unique_join_result_rows,
                                            new_t_full, set_cmp()) - new_t_full;


        Entity *new_t_delta;
        checkCuda(cudaMalloc((void **) &new_t_delta, unique_join_result_rows * sizeof(Entity)));
        t_delta_rows = thrust::set_difference(
                thrust::device,
                new_t_full, new_t_full + new_t_full_rows,
                t_full, t_full + t_full_rows,
                new_t_delta, set_cmp()) - new_t_delta;

        cudaFree(t_delta);
        checkCuda(cudaMalloc((void **) &t_delta, t_delta_rows * sizeof(Entity)));
        thrust::copy(thrust::device,
                     new_t_delta, new_t_delta + t_delta_rows,
                     t_delta);

        cudaFree(t_full);
        checkCuda(cudaMalloc((void **) &t_full, new_t_full_rows * sizeof(Entity)));
        // Copy the deduplicated concatenated result to result
        thrust::copy(thrust::device, new_t_full, new_t_full + new_t_full_rows, t_full);
        t_full_rows = new_t_full_rows;
        cudaFree(new_t_delta);
        cudaFree(join_result);
        cudaFree(offset);
        cudaFree(new_t_full);
        iterations++;
        if (t_delta_rows == 0) {
            break;
        }
    }
    time_point_begin = chrono::high_resolution_clock::now();
    checkCuda(cudaMallocHost((void **) &t_full_host, t_full_rows * sizeof(Entity)));
    cudaMemcpy(t_full_host, t_full, t_full_rows * sizeof(Entity),
               cudaMemcpyDeviceToHost);

    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.union_time += spent_time;
    time_point_begin = chrono::high_resolution_clock::now();
    // Clear memory
    cudaFree(t_delta);
    cudaFree(t_full);
    cudaFree(hash_table);
    cudaFreeHost(t_full_host);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.memory_clear_time += spent_time;
    double calculated_time = output.initialization_time +
                             output.read_time + output.reverse_time + output.hashtable_build_time + output.join_time +
                             output.projection_time +
                             output.union_time + output.deduplication_time + output.memory_clear_time;
    cout << "| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |" << endl;
    cout << "| --- | --- | --- | --- | --- | --- |" << endl;
    cout << "| " << dataset_name << " | " << relation_rows << " | " << t_full_rows;
    cout << fixed << " | " << iterations << " | ";
    cout << fixed << grid_size << " x " << block_size << " | " << calculated_time << " |" << endl;
    output.block_size = block_size;
    output.grid_size = grid_size;
    output.input_rows = relation_rows;
    output.load_factor = load_factor;
    output.hashtable_rows = hash_table_rows;
    output.dataset_name = dataset_name;
    output.total_time = calculated_time;
}

void run_benchmark(int grid_size, int block_size, double load_factor) {
    // Variables to store device information
    int device_id;
    int number_of_sm;

    // Get the current CUDA device
    cudaGetDevice(&device_id);
    // Get the number of streaming multiprocessors (SM) on the device
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, device_id);

    // Set locale for printing numbers with commas as thousands separator
    std::locale loc("");
    std::cout.imbue(loc);
    std::cout << std::fixed;
    std::cout << std::setprecision(4);

    // Separator character for dataset names and paths
    char separator = '\t';

    // Array of dataset names and paths, filename pattern: data_<number_of_rows>.txt
    string datasets[] = {
            "DATA_5", "data/data_5.txt",
            "DATA_6", "data/data_6.txt",
            "OL.cedge_initial", "data/data_7035.txt",
//            "CA-HepTh", "data/data_51971.txt",
//            "SF.cedge", "data/data_223001.txt",
//            "ego-Facebook", "data/data_88234.txt",
//            "wiki-Vote", "data/data_103689.txt",
//            "p2p-Gnutella09", "data/data_26013.txt",
//            "p2p-Gnutella04", "data/data_39994.txt",
//            "cal.cedge", "data/data_21693.txt",
//            "TG.cedge", "data/data_23874.txt",
//            "OL.cedge", "data/data_7035.txt",
//            "luxembourg_osm", "data/data_119666.txt",
//            "fe_sphere", "data/data_49152.txt",
//            "fe_body", "data/data_163734.txt",
//            "cti", "data/data_48232.txt",
//            "fe_ocean", "data/data_409593.txt",
            "wing", "data/data_121544.txt",
            "loc-Brightkite", "data/data_214078.txt",
//            "delaunay_n16", "data/data_196575.txt",
//            "usroads", "data/data_165435.txt",
    };

    // Iterate over the datasets array
    // Each iteration processes a dataset
    for (int i = 0; i < sizeof(datasets) / sizeof(datasets[0]); i += 2) {
        const char *data_path, *dataset_name;
        // Extract the dataset name and path from the array
        dataset_name = datasets[i].c_str();
        data_path = datasets[i + 1].c_str();

        // Get the row size of the dataset
        long int row_size = get_row_size(data_path);

        // Print benchmark information for the current dataset
        cout << "Benchmark for " << dataset_name << endl;
        cout << "----------------------------------------------------------" << endl;

        // Run the GPU graph processing function with the dataset parameters
        gpu_tc(data_path, separator,
               row_size, load_factor,
               grid_size, block_size, dataset_name, number_of_sm);

        cout << endl;
    }
}


int main() {
    run_benchmark(0, 0, 0.4);
    return 0;
}

/*
Run instructions:
make run
*/
