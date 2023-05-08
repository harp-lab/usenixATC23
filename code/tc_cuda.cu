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
    int *relation;
    int *relation_host;
    Entity *hash_table, *result, *t_delta;
    Entity *result_host;
    long int join_result_rows;
    long int t_delta_rows = relation_rows;
    long int result_rows = relation_rows;
    long int iterations = 0;
    long int hash_table_rows = (long int) relation_rows / load_factor;
    hash_table_rows = pow(2, ceil(log(hash_table_rows) / log(2)));

    checkCuda(cudaMallocHost((void **) &relation_host, relation_rows * relation_columns * sizeof(int)));
    checkCuda(cudaMalloc((void **) &relation, relation_rows * relation_columns * sizeof(int)));
    checkCuda(cudaMalloc((void **) &result, result_rows * sizeof(Entity)));
    checkCuda(cudaMalloc((void **) &t_delta, relation_rows * sizeof(Entity)));
    checkCuda(cudaMalloc((void **) &hash_table, hash_table_rows * sizeof(Entity)));
//    checkCuda(cudaMemPrefetchAsync(relation, relation_rows * relation_columns * sizeof(int), device_id));
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
    time_point_begin = chrono::high_resolution_clock::now();
    get_relation_from_file_gpu(relation_host, data_path,
                               relation_rows, relation_columns, separator);
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
    spent_time = timer.get_spent_time();
//    cout << "Hash table build time: " << spent_time << endl;
    output.hashtable_build_time = spent_time;
    output.hashtable_build_rate = (double) relation_rows / spent_time;
    output.join_time += spent_time;

    timer.start_timer();
    // initial result and t delta both are same as the input relation
    initialize_result_t_delta<<<grid_size, block_size>>>(result, t_delta, relation, relation_rows, relation_columns);
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    spent_time = timer.get_spent_time();
    output.union_time += spent_time;
    temp_time_begin = chrono::high_resolution_clock::now();
    thrust::stable_sort(thrust::device, result, result + relation_rows, cmp());
    temp_time_end = chrono::high_resolution_clock::now();
    temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
    sort_time += temp_spent_time;
    output.deduplication_time += temp_spent_time;

    time_point_begin = chrono::high_resolution_clock::now();
    cudaFree(relation);
    cudaFreeHost(relation_host);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.memory_clear_time += spent_time;

#ifdef DEBUG
    cout << "| Iteration | #T Delta | #Join | #Deduplicated join | #Union | #Deduplicated Union | ";
    cout << "Join(s) | Union(s) | Deduplication(s) | Memory clear(s)|"<< endl;
    cout << "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |" << endl;
#endif
    while (true) {
        double temp_join = 0.0, temp_union = 0.0, temp_deduplication = 0.0, temp_memory_clear = 0.0;
        double temp_merge = 0.0, temp_sort = 0.0, temp_unique = 0.0;
#ifdef DEBUG
        cout << "| " << iterations+1 << " | "<< t_delta_rows << " | ";
#endif
        time_point_begin = chrono::high_resolution_clock::now();
        int *offset;
        Entity *join_result;
        checkCuda(cudaMalloc((void **) &offset, t_delta_rows * sizeof(int)));
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_join += spent_time;
        output.join_time += spent_time;
        timer.start_timer();
        get_join_result_size<<<grid_size, block_size>>>(hash_table, hash_table_rows, t_delta, t_delta_rows,
                                                        offset);
        checkCuda(cudaDeviceSynchronize());
        timer.stop_timer();
        spent_time = timer.get_spent_time();
        temp_join += spent_time;
        output.join_time += spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        join_result_rows = thrust::reduce(thrust::device, offset, offset + t_delta_rows, 0);
        thrust::exclusive_scan(thrust::device, offset, offset + t_delta_rows, offset);
        checkCuda(cudaMalloc((void **) &join_result, join_result_rows * sizeof(Entity)));
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_join += spent_time;
        output.join_time += spent_time;
        timer.start_timer();
        get_join_result<<<grid_size, block_size>>>(hash_table, hash_table_rows,
                                                   t_delta, t_delta_rows, offset, join_result);
        checkCuda(cudaDeviceSynchronize());
        timer.stop_timer();
        spent_time = timer.get_spent_time();
        temp_join += spent_time;
        output.join_time += spent_time;
#ifdef DEBUG
        cout << join_result_rows << " | ";
#endif
        // deduplication of projection
        // first sort the array and then remove consecutive duplicated elements
        temp_time_begin = chrono::high_resolution_clock::now();
        thrust::stable_sort(thrust::device, join_result, join_result + join_result_rows, cmp());
        temp_time_end = chrono::high_resolution_clock::now();
        temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
        temp_sort += temp_spent_time;
        temp_deduplication += temp_spent_time;
        sort_time += temp_spent_time;
        output.deduplication_time += temp_spent_time;
        temp_time_begin = chrono::high_resolution_clock::now();
        long int projection_rows = (thrust::unique(thrust::device,
                                                   join_result, join_result + join_result_rows,
                                                   is_equal())) - join_result;
        temp_time_end = chrono::high_resolution_clock::now();
        temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
        temp_unique += temp_spent_time;
        temp_deduplication += temp_spent_time;
        unique_time += temp_spent_time;
        output.deduplication_time += temp_spent_time;
#ifdef DEBUG
        cout << projection_rows << " | ";
#endif
        // show_entity_array(join_result, projection_rows, "join_result");
        time_point_begin = chrono::high_resolution_clock::now();
        cudaFree(t_delta);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_memory_clear += spent_time;
        output.memory_clear_time += spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        checkCuda(cudaMalloc((void **) &t_delta, projection_rows * sizeof(Entity)));
        thrust::copy(thrust::device, join_result, join_result + projection_rows, t_delta);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_join += spent_time;
        output.join_time += spent_time;

        time_point_begin = chrono::high_resolution_clock::now();
        Entity *concatenated_result;
        long int concatenated_rows = projection_rows + result_rows;
        checkCuda(cudaMalloc((void **) &concatenated_result, concatenated_rows * sizeof(Entity)));
        temp_time_begin = chrono::high_resolution_clock::now();
        // merge two sorted array: previous result and join result
        thrust::merge(thrust::device,
                      result, result + result_rows,
                      join_result, join_result + projection_rows,
                      concatenated_result, cmp());
        temp_time_end = chrono::high_resolution_clock::now();
        temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
        temp_merge += temp_spent_time;
        merge_time += temp_spent_time;
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_union += spent_time;
        output.union_time += spent_time;
#ifdef DEBUG
        cout << concatenated_rows << " | ";
#endif
        long int deduplicated_result_rows;
        temp_time_begin = chrono::high_resolution_clock::now();
        deduplicated_result_rows = (thrust::unique(thrust::device,
                                                   concatenated_result,
                                                   concatenated_result + concatenated_rows,
                                                   is_equal())) - concatenated_result;
        temp_time_end = chrono::high_resolution_clock::now();
        temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
        temp_unique += temp_spent_time;
        unique_time += temp_spent_time;
        temp_deduplication += temp_spent_time;
        output.deduplication_time += temp_spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        cudaFree(result);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_memory_clear += spent_time;
        output.memory_clear_time += spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        checkCuda(cudaMalloc((void **) &result, deduplicated_result_rows * sizeof(Entity)));
        // Copy the deduplicated concatenated result to result
        thrust::copy(thrust::device, concatenated_result,
                     concatenated_result + deduplicated_result_rows, result);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_union += spent_time;
        output.union_time += spent_time; // changed this time from deduplication to union
#ifdef DEBUG
        cout << deduplicated_result_rows << " | ";
#endif
        t_delta_rows = projection_rows;
        time_point_begin = chrono::high_resolution_clock::now();
        cudaFree(join_result);
        cudaFree(offset);
        cudaFree(concatenated_result);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_memory_clear += spent_time;
        output.memory_clear_time += spent_time;
#ifdef DEBUG
        cout << temp_join << " | ";
        cout << temp_union << " | ";
        cout << temp_deduplication << " | ";
        cout << temp_memory_clear << " | " << endl;
#endif

        if (result_rows == deduplicated_result_rows) {
            iterations++;
            break;
        }
        result_rows = deduplicated_result_rows;
        iterations++;
    }
//    show_entity_array(result, result_rows, "Result");
    time_point_begin = chrono::high_resolution_clock::now();
    checkCuda(cudaMallocHost((void **) &result_host, result_rows * sizeof(Entity)));
    cudaMemcpy(result_host, result, result_rows * sizeof(Entity),
               cudaMemcpyDeviceToHost);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.union_time += spent_time;
    time_point_begin = chrono::high_resolution_clock::now();
    cudaFree(t_delta);
    cudaFree(result);
    cudaFree(hash_table);
    cudaFreeHost(result_host);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.memory_clear_time += spent_time;
    double calculated_time = output.initialization_time +
                             output.read_time + output.reverse_time + output.hashtable_build_time + output.join_time +
                             output.projection_time +
                             output.union_time + output.deduplication_time + output.memory_clear_time;
    cout << endl;
    cout << "| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |" << endl;
    cout << "| --- | --- | --- | --- | --- | --- |" << endl;
    cout << "| " << dataset_name << " | " << relation_rows << " | " << result_rows;
    cout << fixed << " | " << iterations << " | ";
    cout << fixed << grid_size << " x " << block_size << " | " << calculated_time << " |\n" << endl;
    output.block_size = block_size;
    output.grid_size = grid_size;
    output.input_rows = relation_rows;
    output.load_factor = load_factor;
    output.hashtable_rows = hash_table_rows;
    output.dataset_name = dataset_name;
    output.total_time = calculated_time;

    cout << endl;
    cout << "Initialization: " << output.initialization_time;
    cout << ", Read: " << output.read_time << endl;
    cout << "Hashtable rate: " << output.hashtable_build_rate << " keys/s, time: ";
    cout << output.hashtable_build_time << endl;
    cout << "Join: " << output.join_time << endl;
    cout << "Deduplication: " << output.deduplication_time;
    cout << " (sort: " << sort_time << ", unique: " << unique_time << ")" << endl;
    cout << "Memory clear: " << output.memory_clear_time << endl;
    cout << "Union: " << output.union_time << " (merge: " << merge_time << ")" << endl;
    cout << "Total: " << output.total_time << endl;
}


void run_benchmark(int grid_size, int block_size, double load_factor) {
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, device_id);
    std::locale loc("");
    std::cout.imbue(loc);
    std::cout << std::fixed;
    std::cout << std::setprecision(4);
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
            "usroads-48", "../data/data_161950.txt",
            "String 9990", "../data/data_9990.txt",
            "String 2990", "../data/data_2990.txt",
            "string 4", "../data/data_4.txt",
            "talk 5", "../data/data_5.txt",
            "cyclic 3", "../data/data_3.txt",
    };
    for (int i = 0; i < sizeof(datasets) / sizeof(datasets[0]); i += 2) {
        const char *data_path, *dataset_name;
        dataset_name = datasets[i].c_str();
        data_path = datasets[i + 1].c_str();
        long int row_size = get_row_size(data_path);
        cout << "Benchmark for " << dataset_name << endl;
        cout << "----------------------------------------------------------" << endl;
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

// Benchmark
// nvcc tc_cuda.cu -run -o tc_cuda.out
// or
// make run
