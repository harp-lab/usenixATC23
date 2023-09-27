// #include <cstdio>
// #include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>
#include <iomanip>
#include <assert.h>
// #include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/set_operations.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

// 
// #include "common/error_handler.cu"
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

struct KernelTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    KernelTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~KernelTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void start_timer() {
        cudaEventRecord(start, 0);
    }

    void stop_timer() {
        cudaEventRecord(stop, 0);
    }

    float get_spent_time() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        elapsed /= 1000.0;
        return elapsed;
    }
};

struct Output {
    int block_size;
    int grid_size;
    long int input_rows;
    long int hashtable_rows;
    double load_factor;
    double initialization_time;
    double memory_clear_time;
    double read_time;
    double reverse_time;
    double hashtable_build_time;
    long int hashtable_build_rate;
    double join_time;
    double projection_time;
    double deduplication_time;
    double union_time;
    double total_time;
    const char *dataset_name;
} output;

/*
 * Method that returns position in the hashtable for a key using Murmur3 hash
 * */


using u64 = unsigned long long;
using u32 = unsigned long;

using column_type = u64;
using tuple_type = column_type*;

// TODO: use thrust vector as tuple type??
// using t_gpu_index = thrust::device_vector<u64>;
// using t_gpu_tuple = thrust::device_vector<u64>;

// using t_data_internal = thrust::device_vector<u64>;
/**
 * @brief u64* to store the actual relation tuples, for serialize concern
 * 
 */
using t_data_internal = u64*;

/**
 * @brief TODO: remove this use comparator function
 * 
 * @param t1 
 * @param t2 
 * @param l 
 * @return true 
 * @return false 
 */
 __host__
 __device__
inline bool tuple_eq(tuple_type t1, tuple_type t2, u64 l) {
    for (int i = 0; i < l; i++) {
        if (t1[i] != t2[i]) {
            return false;
        }
    }
    return true;
}

struct t_equal {
    u64 arity;

    t_equal(u64 arity) {
        this->arity = arity;
    }

    __host__ __device__
    bool operator()(const tuple_type &lhs, const tuple_type &rhs) {
        for (int i = 0; i < arity; i++) {
            if (lhs[i] != rhs[i]) {
                return false;
            }
        }
        return true;
    }
};

/**
 * @brief fnv1-a hash used in original slog backend
 * 
 * @param start_ptr 
 * @param prefix_len 
 * @return __host__ __device__
 */
__host__
__device__
inline u64 prefix_hash(tuple_type start_ptr, u64 prefix_len)
{
    const u64 base = 14695981039346656037ULL;
    const u64 prime = 1099511628211ULL;

    u64 hash = base;
    for (u64 i = 0; i < prefix_len; ++i)
    {
        u64 chunk = start_ptr[i];
        hash ^= chunk & 255ULL;
        hash *= prime;
        for (char j = 0; j < 7; ++j)
        {
            chunk = chunk >> 8;
            hash ^= chunk & 255ULL;
            hash *= prime;
        }
    }
    return hash;
}

// change to std
struct tuple_indexed_less {

    // u64 *index_columns;
    u64 index_column_size;
    int arity;
    
    tuple_indexed_less(u64 index_column_size, int arity) {
        // this->index_columns = index_columns;
        this->index_column_size = index_column_size;
        this->arity = arity;
    }

    __host__ __device__
    bool operator()(const tuple_type &lhs, const tuple_type &rhs) {
        // fetch the index
        // compare hash first, could be index very different but share the same hash
        if (prefix_hash(lhs, index_column_size) == prefix_hash(rhs, index_column_size)) {
            // same hash
            for (u64 i = 0; i < arity; i++) {
                if (lhs[i] < rhs[i]) {
                    return true;
                } else if (lhs[i] > rhs[i]) {
                    return false;
                }
            }
            return false;
        } else if (prefix_hash(lhs, index_column_size) < prefix_hash(rhs, index_column_size)) {
            return true;
        } else {
            return false;
        }
    }
};

struct tuple_weak_greater {

    int arity;
    
    tuple_weak_greater(int arity) {
        this->arity = arity;
    }

    __host__ __device__
    bool operator()(const tuple_type &lhs, const tuple_type &rhs) {

        for (u64 i = 0; i < arity; i++) {
            if (lhs[i] > rhs[i]) {
                return true;
            } else if (lhs[i] < rhs[i]) {
                return false;
            }
        }
        return false;
    };
};
struct tuple_weak_less {

    int arity;
    
    tuple_weak_less(int arity) {
        this->arity = arity;
    }

    __host__ __device__
    bool operator()(const tuple_type &lhs, const tuple_type &rhs) {

        for (u64 i = 0; i < arity; i++) {
            if (lhs[i] < rhs[i]) {
                return true;
            } else if (lhs[i] > rhs[i]) {
                return false;
            }
        }
        return false;
    };
};

/**
 * @brief A hash table entry
 * TODO: no need for struct actually, a u64[2] should be enough, easier to init
 * 
 */
struct MEntity {
    // index position in actual index_arrary
    u64 key;
    // tuple position in actual data_arrary
    u64 value;
};

#define EMPTY_HASH_ENTRY ULLONG_MAX
/**
 * @brief a C-style hashset indexing based relation container.
 *        Actual data is still stored using sorted set.
 *        Different from normal btree relation, using hash table storing the index to accelarte
 *        range fetch.
 *        Good:
 *           - fast range fetch, in Shovon's ATC paper it shows great performance.
 *           - fast serialization, its very GPU friendly and also easier for MPI inter-rank comm
 *             transmission.
 *        Bad:
 *           - need reconstruct index very time tuple is inserted (need more reasonable algorithm).
 *           - sorting is a issue, each update need resort everything seems stupid.
 * 
 */
struct GHashRelContainer {
    // open addressing hashmap for indexing
    MEntity* index_map;
    u64 index_map_size;
    float index_map_load_factor;

    // index prefix length
    // don't have to be u64,int is enough
    // u64 *index_columns;
    u64 index_column_size;

    // the pointer to flatten tuple, all tuple pointer here need to be sorted 
    tuple_type* tuples;
    // flatten tuple data
    column_type* data_raw;
    // number of tuples
    u64 tuple_counts;
    // actual tuple rows in flatten data, this maybe different from
    // tuple_counts when deduplicated
    u64 data_raw_row_size;
    int arity;
};

// kernels

/**
 * @brief fill in index hash table for a relation in parallel, assume index is correctly initialized, data has been loaded
 *        , deduplicated and sorted
 * 
 * @param target the hashtable to init
 * @return dedeuplicated_bitmap 
 */
__global__
void calculate_index_hash(GHashRelContainer* target) {
    u64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= target->tuple_counts) return;

    u64 stride = blockDim.x * gridDim.x;

    for (u64 i = index; i < target->tuple_counts; i += stride) {
        tuple_type cur_tuple = target->tuples[i];

        u64 hash_val = prefix_hash(cur_tuple, target->index_column_size);
        u64 request_hash_index = hash_val % target->index_map_size;
        u64 position = request_hash_index;
        // insert into data container
        while (true) {
            // critical condition!
            u64 existing_key = atomicCAS(&(target->index_map[position].key), EMPTY_HASH_ENTRY, hash_val);
            u64 existing_value = target->index_map[position].value;
            if (existing_key == EMPTY_HASH_ENTRY || existing_key == hash_val) {
                while (true) {
                    if (existing_value <= i) {
                        break;
                    } else {
                        // need swap
                        existing_value = atomicCAS(&(target->index_map[position].value), existing_value, i);
                    }
                }
                break;
            }
            
            position = (position + 1) % target->index_map_size;
        }
    }
}

/**
 * @brief count how many non empty hash entry in index map
 * 
 * @param target target relation hash table
 * @param size return the size
 * @return __global__ 
 */
__global__
void count_index_entry_size(GHashRelContainer* target, u64* size) {
    u64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= target->index_map_size) return;

    u64 stride = blockDim.x * gridDim.x;
    for (u64 i = index; i < target->index_map_size; i += stride) {
        if (target->index_map[i].value != EMPTY_HASH_ENTRY) {
            atomicAdd(size, 1);
        }
    }
}

/**
 * @brief rehash to make index map more compact, the new index hash size is already update in target
 *        new index already inited to empty table and have new size.
 * 
 * @param target 
 * @param old_index_map index map before compaction
 * @param old_index_map_size original size of index map before compaction
 * @return __global__ 
 */
__global__
void shrink_index_map(GHashRelContainer* target, MEntity* old_index_map, u64 old_index_map_size) {
    u64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= old_index_map_size) return;

    u64 stride = blockDim.x * gridDim.x;
    for (u64 i = index; i < old_index_map_size; i += stride) {
        if (target->index_map[i].value != EMPTY_HASH_ENTRY) {
            u64 hash_val = target->index_map[i].key;
            u64 position = hash_val % target->index_map_size;
            while(true) {
                u64 existing_key = atomicCAS(&target->index_map[position].key, EMPTY_HASH_ENTRY, hash_val);
                if (existing_key == EMPTY_HASH_ENTRY) {
                    target->index_map[position].key = hash_val;
                    break;
                } else if(existing_key == hash_val) {
                    // hash for tuple's index column has already been recorded
                    break;
                }
                position = (position + 1) % target->index_map_size;
            }
        }
    }
}


// NOTE: must copy size out of gpu kernal code!!!
/**
 * @brief acopy the **index** from a relation to another, please use this together with *copy_data*, and settle up all metadata before copy
 * 
 * @param source source relation
 * @param destination destination relation
 * @return __global__ 
 */
__global__
void acopy_entry(GHashRelContainer* source, GHashRelContainer* destination) {
    auto source_rows = source->index_map_size;
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= source_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < source_rows; i += stride) {
        destination->index_map[i].key = source->index_map[i].key;
        destination->index_map[i].value = source->index_map[i].value;
    }
}
__global__
void acopy_data(GHashRelContainer *source, GHashRelContainer *destination) {
    auto data_rows = source->tuple_counts;
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= data_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < data_rows; i += stride) {
        tuple_type cur_src_tuple = source->tuples[i]; 
        for (int j = 0; j < source->arity; j++) {
            destination->data_raw[i*source->arity+j] = cur_src_tuple[j];
        }
        destination->tuples[i] = destination->tuples[i*source->arity];
    }
}

// 
/**
 * @brief a CUDA kernel init the index entry map of a hashtabl
 * 
 * @param target the hashtable to init
 * @return void 
 */
__global__
void init_index_map(GHashRelContainer* target) {
    auto source = target->index_map;
    auto source_rows = target->index_map_size;
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= source_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (u64 i = index; i < source_rows; i += stride) {
        source[i].key = EMPTY_HASH_ENTRY;
        source[i].value = EMPTY_HASH_ENTRY;
    }
}

// a helper function to init an unsorted tuple arrary from raw data
__global__
void init_tuples_unsorted(tuple_type* tuples, column_type* raw_data, int arity, u64 rows) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= rows) return;

    int stride = blockDim.x * gridDim.x;
    for (u64 i = index; i < rows; i += stride) {
        tuples[i] = raw_data + i * arity;
    }
}

/**
 * @brief for all tuples in outer table, match same prefix with inner table
 *
 * @note can we use pipeline here? since many matching may acually missing
 * 
 * @param inner_table the hashtable to iterate
 * @param outer_table the hashtable to match
 * @param join_column_counts number of join columns (inner and outer must agree on this)
 * @param  return value stored here, size of joined tuples
 * @return void 
 */
__global__
void get_join_result_size(GHashRelContainer* inner_table,
                          GHashRelContainer* outer_table,
                          int join_column_counts,
                          u64* join_result_size,
                          u64* debug = nullptr) {
    u64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= outer_table->tuple_counts) return;
    u64 stride = blockDim.x * gridDim.x;

    for (u64 i = index; i < outer_table->tuple_counts; i += stride) {
        tuple_type outer_tuple = outer_table->tuples[i];

        // column_type* outer_indexed_cols;
        // cudaMalloc((void**) outer_indexed_cols, outer_table->index_column_size * sizeof(column_type));
        // for (size_t idx_i = 0; idx_i < outer_table->index_column_size; idx_i ++) {
        //     outer_indexed_cols[idx_i] = outer_table->tuples[i * outer_table->arity][outer_table->index_columns[idx_i]];
        // }
        u64 current_size = 0;
        join_result_size[i] = 0;
        u64 hash_val = prefix_hash(outer_tuple, outer_table->index_column_size);
        // the index value "pointer" position in the index hash table 
        u64 index_position = hash_val % inner_table->index_map_size;
        // 64 bit hash is less likely to have collision
        // partially solve hash conflict? maybe better for performance
        bool hash_not_exists = false;
        while (true) {
            if (inner_table->index_map[index_position].key == hash_val) {
                break;
            } else if (inner_table->index_map[index_position].key == EMPTY_HASH_ENTRY) {
                hash_not_exists = true;
                break;
            }
            index_position = (index_position + 1) % inner_table->index_map_size;
        }
        if (hash_not_exists) {
            continue;
        }
        // pull all joined elements
        u64 position = inner_table->index_map[index_position].value;
        while (true) {
            tuple_type cur_inner_tuple = inner_table->tuples[position];
            bool cmp_res = tuple_eq(inner_table->tuples[position],
                                    outer_tuple,
                                    join_column_counts);
            // if (outer_tuple[0] == 6662) {
            //     printf("wwwwwwwwwwwwwwwwwwwwww %lld outer: %lld, %lld; inner: %lld, %lld;\n",
            //            position,
            //            outer_tuple[0], outer_tuple[1],
            //            cur_inner_tuple[0], cur_inner_tuple[1]);
            // }
            if (cmp_res) {
                current_size++;
            }else {
                
                u64 inner_tuple_hash = prefix_hash(cur_inner_tuple, inner_table->index_column_size);
                if (inner_tuple_hash != hash_val) {
                    // bucket end
                    break;
                }
                // collision, keep searching
            }
            position = position + 1;
            if (position > inner_table->tuple_counts - 1) {
                // end of data arrary
                break;
            }
        }
        join_result_size[i] = current_size;
        // cudaFree(outer_indexed_cols);
    }
}

/**
 * @brief compute the join result
 * 
 * @param inner_table 
 * @param outer_table 
 * @param join_column_counts 
 * @param output_reorder_array reorder array for output relation column selection, arrary pos < inner->arity is index in inner, > is index in outer.
 * @param output_arity output relation arity
 * @param output_raw_data join result, need precompute the size
 * @return __global__ 
 */
__global__
void get_join_result(GHashRelContainer* inner_table,
                     GHashRelContainer* outer_table,
                     int join_column_counts,
                     int* output_reorder_array,
                     int output_arity,
                     column_type* output_raw_data,
                     u64* res_count_array,
                     u64* res_offset) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= outer_table->tuple_counts) return;

    int stride = blockDim.x * gridDim.x;

    for (u64 i = index; i < outer_table->tuple_counts; i += stride) {
        if (res_count_array[i] == 0) {
            continue;
        }
        u64 tuple_raw_pos = i*((u64)outer_table->arity);
        tuple_type outer_tuple = outer_table->tuples[i];

        // column_type* outer_indexed_cols;
        // cudaMalloc((void**) outer_indexed_cols, outer_table->index_column_size * sizeof(column_type));
        // for (size_t idx_i = 0; idx_i < outer_table->index_column_size; idx_i ++) {
        //     outer_indexed_cols[idx_i] = outer_table->tuples[i * outer_table->arity][outer_table->index_columns[idx_i]];
        // }
        int current_new_tuple_cnt = 0;
        u64 hash_val = prefix_hash(outer_tuple, outer_table->index_column_size);
        // the index value "pointer" position in the index hash table 
        u64 index_position = hash_val % inner_table->index_map_size;
        bool hash_not_exists = false;
        while (true) {
            if (inner_table->index_map[index_position].key == hash_val) {
                break;
            } else if (inner_table->index_map[index_position].key == EMPTY_HASH_ENTRY) {
                hash_not_exists = true;
                break;
            }
            index_position = (index_position + 1) % inner_table->index_map_size;
        }
        if (hash_not_exists) {
            continue;
        }

        // pull all joined elements
        u64 position = inner_table->index_map[index_position].value;
        while (true) {
            // TODO: always put join columns ahead? could be various benefits but memory is issue to mantain multiple copies
            bool cmp_res = tuple_eq(inner_table->tuples[position],
                                    outer_tuple,
                                    join_column_counts);
            if (cmp_res) {
                // tuple prefix match, join here
                tuple_type inner_tuple = inner_table->tuples[position];
                tuple_type new_tuple = output_raw_data + (res_offset[i] + current_new_tuple_cnt) * output_arity;

                for (int j = 0; j < output_arity; j++) {
                    if (output_reorder_array[j] < inner_table->arity) {
                        new_tuple[j] = inner_tuple[output_reorder_array[j]];
                    } else {
                        new_tuple[j] = outer_tuple[output_reorder_array[j] - inner_table->arity];
                    }
                }
                current_new_tuple_cnt++;
                if (current_new_tuple_cnt >= res_count_array[i]) {
                    return;
                }
            }else {
                // if not prefix not match, there might be hash collision
                tuple_type cur_inner_tuple = inner_table->tuples[position];
                u64 inner_tuple_hash = prefix_hash(cur_inner_tuple, inner_table->index_column_size);
                if (inner_tuple_hash != hash_val) {
                    // bucket end
                    break;
                }
                // collision, keep searching
            }
            position = position + 1;
            if (position > (inner_table->tuple_counts - 1)) {
                // end of data arrary
                break;
            }
        }
    }
}

__global__
void flatten_tuples_raw_data(tuple_type* tuple_pointers, column_type* raw,
                    u64 tuple_counts, int arity) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= tuple_counts) return;

    int stride = blockDim.x * gridDim.x;
    for (u64 i = index; i < tuple_counts; i += stride) {
        for (int j = 0; j < arity; j++) {
            raw[i*arity+j] = tuple_pointers[i][j];
        }
    }
}

///////////////////////////////////////////////////////
// test helper

void print_hashes(GHashRelContainer* target, const char *rel_name) {
    MEntity* host_map;
    cudaMallocHost((void**) &host_map, target->index_map_size * sizeof(MEntity));
    cudaMemcpy(host_map, target->index_map, target->index_map_size * sizeof(MEntity),
               cudaMemcpyDeviceToHost);
    std::cout << "Relation hash >>> " << rel_name << std::endl;
    for (u64 i = 0; i < target->index_map_size; i++) {
        std::cout << host_map[i].key << "    " << host_map[i].value << std::endl;
    }
    std::cout << "end <<<" << std::endl;
    cudaFreeHost(host_map);
}

void print_tuple_rows(GHashRelContainer* target, const char *rel_name) {
    // sort first
    tuple_type* natural_ordered;
    cudaMalloc((void**) &natural_ordered, target->tuple_counts * sizeof(tuple_type));
    cudaMemcpy(natural_ordered, target->tuples, target->tuple_counts * sizeof(tuple_type),
               cudaMemcpyDeviceToDevice);
    thrust::sort(thrust::device, natural_ordered, natural_ordered+target->tuple_counts,
                 tuple_weak_less(target->arity));

    tuple_type* tuples_host;
    cudaMallocHost((void**) &tuples_host, target->tuple_counts * sizeof(tuple_type));
    cudaMemcpy(tuples_host, natural_ordered, target->tuple_counts * sizeof(tuple_type),
               cudaMemcpyDeviceToHost);
    std::cout << "Relation tuples >>> " << rel_name << std::endl;
    std::cout << "Total tuples counts:  " <<  target->tuple_counts << std::endl;
    for (u64 i = 0; i < target->tuple_counts; i++) {
        tuple_type cur_tuple = tuples_host[i];
        tuple_type cur_tuple_host;
        cudaMallocHost((void**) &cur_tuple_host, target->arity * sizeof(column_type));
        cudaMemcpy(cur_tuple_host, cur_tuple, target->arity * sizeof(column_type),
                   cudaMemcpyDeviceToHost);
        for (int j = 0; j < target->arity; j++) {

            std::cout << cur_tuple_host[j] << "\t";
        }
        std::cout << std::endl;
        cudaFreeHost(cur_tuple_host);
    }
    std::cout << "end <<<" << std::endl;

    cudaFreeHost(tuples_host);
    cudaFree(natural_ordered);
}

void print_tuple_raw_data(GHashRelContainer* target, const char *rel_name) {
    column_type* raw_data_host;
    u64 mem_raw = target->data_raw_row_size * target->arity * sizeof(column_type);
    cudaMallocHost((void**) &raw_data_host, mem_raw);
    cudaMemcpy(raw_data_host, target->data_raw, mem_raw, cudaMemcpyDeviceToHost);
    std::cout << "Relation raw tuples >>> " << rel_name << std::endl;
    std::cout << "Total raw tuples counts:  " <<  target->data_raw_row_size << std::endl;
    for (u64 i = 0; i < target->data_raw_row_size; i++) {
        for (int j = 0; j < target->arity; j++) {
            std::cout << raw_data_host[i*target->arity + j] << "    ";
        }
        std::cout << std::endl;
    }
    cudaFreeHost(raw_data_host);
}

//////////////////////////////////////////////////////
// CPU functions

/**
 * @brief load raw data into relation container
 * 
 * @param target hashtable struct in host
 * @param arity 
 * @param data raw data on host
 * @param data_row_size 
 * @param index_columns index columns id in host
 * @param index_column_size 
 * @param index_map_load_factor 
 * @param grid_size 
 * @param block_size
 * @param gpu_data_flag if data is a GPU memory address directly assign to target's data_raw
 */
void load_relation(GHashRelContainer* target, int arity, column_type* data, u64 data_row_size,
                   u64 index_column_size, float index_map_load_factor,
                   int grid_size, int block_size,
                   bool gpu_data_flag=false, bool sorted_flag=false) {
    target->arity = arity;
    target->tuple_counts = data_row_size;
    target->data_raw_row_size = data_row_size;
    target->index_map_load_factor = index_map_load_factor;
    target->index_column_size = index_column_size;
    // load index selection into gpu
    // u64 index_columns_mem_size = index_column_size * sizeof(u64);
    // checkCuda(cudaMalloc((void**) &(target->index_columns), index_columns_mem_size));
    // cudaMemcpy(target->index_columns, index_columns, index_columns_mem_size, cudaMemcpyHostToDevice);
    if (data_row_size == 0) {
        return;
    }
    // load raw data from host
    if (gpu_data_flag) {
        target->data_raw = data;
    } else {
        u64 relation_mem_size = data_row_size * ((u64)arity) * sizeof(column_type);
        checkCuda(
            cudaMalloc((void **)&(target->data_raw), relation_mem_size)
        );
        cudaMemcpy(target->data_raw, data, relation_mem_size, cudaMemcpyHostToDevice);
    }
    // init tuple to be unsorted raw tuple data address
    checkCuda(cudaMalloc((void**) &target->tuples, data_row_size * sizeof(tuple_type)));
    init_tuples_unsorted<<<grid_size, block_size>>>(target->tuples, target->data_raw, arity, data_row_size);
    // sort raw data
    if (!sorted_flag) {
        thrust::sort(thrust::device, target->tuples, target->tuples+data_row_size,
                            tuple_indexed_less(index_column_size, arity));
        // print_tuple_rows(target, "after sort");
        
        // deduplication here?
        tuple_type* new_end = thrust::unique(thrust::device, target->tuples, target->tuples+data_row_size,
                                            t_equal(arity));    
        data_row_size = new_end - target->tuples;
    }

    target->tuple_counts = data_row_size;
    // print_tuple_rows(target, "after dedup");

    // init the index map
    // set the size of index map same as data, (this should give us almost no conflict)
    // however this can be memory inefficient
    target->index_map_size = std::ceil(data_row_size / index_map_load_factor);
    // target->index_map_size = data_row_size;
    u64 index_map_mem_size = target->index_map_size * sizeof(MEntity);
    checkCuda(
        cudaMalloc((void**)&(target->index_map), index_map_mem_size)
    );
    
    // load inited data struct into GPU memory
    GHashRelContainer* target_device;
    checkCuda(cudaMalloc((void**) &target_device, sizeof(GHashRelContainer)));
    cudaMemcpy(target_device, target, sizeof(GHashRelContainer), cudaMemcpyHostToDevice);
    init_index_map<<<grid_size, block_size>>>(target_device);
    // std::cout << "finish init index map" << std::endl;
    // print_hashes(target, "after construct index map");
    // calculate hash 
    calculate_index_hash<<<grid_size, block_size>>>(target_device);
    cudaFree(target_device);
}

/**
 * @brief copy a relation into an **empty** relation
 * 
 * @param dst 
 * @param src 
 */
void copy_relation_container(GHashRelContainer* dst, GHashRelContainer* src) {
    dst->index_map_size = src->index_map_size;
    dst->index_map_load_factor = src->index_map_load_factor;
    checkCuda(cudaMalloc((void**) &dst->index_map, dst->index_map_size*sizeof(MEntity)));
    cudaMemcpy(dst->index_map, src->index_map,
               dst->index_map_size*sizeof(MEntity), cudaMemcpyDeviceToDevice);
    dst->index_column_size = src->index_column_size;

    dst->tuple_counts = src->tuple_counts;
    dst->data_raw_row_size = src->data_raw_row_size;
    dst->arity = src->arity;
    checkCuda(cudaMalloc((void**) &dst->tuples, dst->tuple_counts*sizeof(tuple_type)));
    cudaMemcpy(dst->tuples, src->tuples,
               src->tuple_counts*sizeof(tuple_type),
               cudaMemcpyDeviceToDevice);
    checkCuda(cudaMalloc((void**) &dst->data_raw,
                         dst->arity * dst->data_raw_row_size * sizeof(column_type)));
    cudaMemcpy(dst->data_raw, src->data_raw,
               dst->arity * dst->data_raw_row_size * sizeof(column_type),
               cudaMemcpyDeviceToDevice);
}

/**
 * @brief clean all data in a relation container
 * 
 * @param target 
 */
void free_relation(GHashRelContainer* target) {
    target->tuple_counts = 0;
    cudaFree(target->index_map);
    cudaFree(target->tuples);
    cudaFree(target->data_raw);
}

/**
 * @brief merge src relation into target relation (they must have same arity)
 * 
 * @param target 
 * @param src 
 */
void merge_relation_container(GHashRelContainer* target, GHashRelContainer* src,
                              int grid_size, int block_size, float* detail_time) {
    assert(target->arity == src->arity);
    KernelTimer timer;
    u64 new_tuple_counts = target->tuple_counts+src->tuple_counts;
    tuple_type* tuple_merge_buffer;
    checkCuda(cudaMalloc((void**) &tuple_merge_buffer, new_tuple_counts*sizeof(tuple_type)));
    timer.start_timer();
    thrust::merge(thrust::device,
                  target->tuples, target->tuples+target->tuple_counts,
                  src->tuples, src->tuples+src->tuple_counts,
                  tuple_merge_buffer,
                  tuple_indexed_less(target->index_column_size, target->arity));
        
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    detail_time[0] = timer.get_spent_time();

    timer.start_timer();
    u64 new_raw_data_mem = new_tuple_counts * target->arity *sizeof(column_type);
    column_type* new_raw_data;
    checkCuda(cudaMalloc((void**) &new_raw_data, new_raw_data_mem));
    flatten_tuples_raw_data<<<grid_size, block_size>>>(tuple_merge_buffer, new_raw_data,
                                                       new_tuple_counts, target->arity);
    checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    detail_time[1] = timer.get_spent_time();

    // cudaMemcpy(new_raw_data, target->data_raw, target_raw_data_mem, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(new_raw_data+target->arity*target->data_raw_row_size,
    //            src->data_raw, src_raw_data_mem, cudaMemcpyDeviceToDevice);
    timer.start_timer();
    free_relation(target);
    load_relation(target, src->arity, new_raw_data, new_tuple_counts, target->index_column_size,
                  target->index_map_load_factor, grid_size, block_size, true, true);
    timer.stop_timer();
    detail_time[2] = timer.get_spent_time();
    cudaFree(tuple_merge_buffer);
}
/**
 * @brief binary join, close to local_join in slog's join RA operator
 * 
 * @param inner 
 * @param outer    
 * @param block_size 
 */
void binary_join(GHashRelContainer* inner, GHashRelContainer* outer,
                 GHashRelContainer* output_newt,
                 int* reorder_array,
                 int reorder_array_size,
                 int grid_size, int block_size) {
    
    checkCuda(cudaDeviceSynchronize());
    GHashRelContainer* inner_device;
    checkCuda(cudaMalloc((void**) &inner_device, sizeof(GHashRelContainer)));
    cudaMemcpy(inner_device, inner, sizeof(GHashRelContainer), cudaMemcpyHostToDevice);
    GHashRelContainer* outer_device;
    checkCuda(cudaMalloc((void**) &outer_device, sizeof(GHashRelContainer)));
    cudaMemcpy(outer_device, outer, sizeof(GHashRelContainer), cudaMemcpyHostToDevice);

    u64* result_counts_array;
    checkCuda(cudaMalloc((void**) &result_counts_array, outer->tuple_counts * sizeof(u64)));

    int* reorder_array_device;
    checkCuda(cudaMalloc((void**) &reorder_array_device, reorder_array_size * sizeof(int)));
    cudaMemcpy(reorder_array_device, reorder_array, reorder_array_size * sizeof(int), cudaMemcpyHostToDevice);
    // print_tuple_rows(outer, "outer");
    
    // std::cout << "inner : " << inner->tuple_counts << " outer: " << outer->tuple_counts << std::endl;
    checkCuda(cudaDeviceSynchronize());
    get_join_result_size<<<grid_size, block_size>>>(inner_device, outer_device, outer->index_column_size, result_counts_array);

    checkCuda(cudaDeviceSynchronize());

    // u64* result_counts_array_host;
    // cudaMallocHost((void**) &result_counts_array_host, outer->tuple_counts * sizeof(u64));
    // cudaMemcpy(result_counts_array_host, result_counts_array, outer->tuple_counts * sizeof(u64), cudaMemcpyDeviceToHost);
    
    u64 total_result_rows = thrust::reduce(thrust::device, result_counts_array, result_counts_array+outer->tuple_counts, 0);

    checkCuda(cudaDeviceSynchronize());
    // std::cout << "join result size(non dedup) " << total_result_rows << std::endl;
    u64* result_counts_offset;
    checkCuda(cudaMalloc((void**) &result_counts_offset, outer->tuple_counts * sizeof(u64)));
    cudaMemcpy(result_counts_offset, result_counts_array, outer->tuple_counts * sizeof(u64), cudaMemcpyDeviceToDevice);
    thrust::exclusive_scan(thrust::device, result_counts_offset, result_counts_offset + outer->tuple_counts, result_counts_offset);

    // u64* result_counts_offset_host;
    // cudaMallocHost((void**) &result_counts_offset_host, outer->tuple_counts * sizeof(u64));
    // cudaMemcpy(result_counts_offset_host, result_counts_offset, outer->tuple_counts * sizeof(u64), cudaMemcpyDeviceToHost);
    // std::cout << "wwwwwwwwwwwww" <<std::endl;
    // for (u64 i = 0; i < outer->tuple_counts; i++) {
    //     std::cout << result_counts_offset_host[i] << std::endl;
    // }
    checkCuda(cudaDeviceSynchronize());

    column_type* join_res_raw_data;
    checkCuda(cudaMalloc((void**) &join_res_raw_data, total_result_rows * output_newt->arity * sizeof(column_type)));
    get_join_result<<<grid_size, block_size>>>(inner_device, outer_device, outer->index_column_size, reorder_array_device, output_newt->arity,
                                               join_res_raw_data, result_counts_array, result_counts_offset);
    checkCuda(cudaDeviceSynchronize());
    // cudaFree(result_counts_array);

    column_type* foobar_raw_data_host;
    cudaMallocHost((void**) &foobar_raw_data_host, total_result_rows * output_newt->arity * sizeof(column_type));
    cudaMemcpy(foobar_raw_data_host, join_res_raw_data, total_result_rows * output_newt->arity * sizeof(column_type),cudaMemcpyDeviceToHost);
    // std::cout << "wwwwwwwwww" << std::endl;
    // for (u64 i = 0; i < total_result_rows; i++) {
    //     for (int j = 0; j < output_newt->arity; j++) {
    //         std::cout << foobar_raw_data_host[i*output_newt->arity + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    checkCuda(cudaDeviceSynchronize());

    // // reload newt
    free_relation(output_newt);
    load_relation(output_newt, output_newt->arity, join_res_raw_data, total_result_rows, 1, 0.6, grid_size, block_size);
    
    // print_tuple_rows(output_newt, "output_newtr");
    checkCuda(cudaDeviceSynchronize());
    // std::cout << "join result size " << output_newt->tuple_counts << std::endl;

    cudaFree(inner_device);
    cudaFree(outer_device);
}

void transitive_closure(GHashRelContainer* edge_2__2_1, GHashRelContainer* path_2__1_2,
                        int block_size, int grid_size) {
    KernelTimer timer;
    // copy struct into gpu
    GHashRelContainer* path_full = path_2__1_2;

    // construct newt/delta for path
    GHashRelContainer* path_newt = new GHashRelContainer();
    column_type foobar_dummy_data[2] = {0,0};
    load_relation(path_newt, 2, foobar_dummy_data, 0, 1, 0.6, grid_size, block_size);
    GHashRelContainer* path_delta = new GHashRelContainer();
    // before first iteration load all full into delta

    copy_relation_container(path_delta, path_2__1_2);   
    // print_tuple_rows(path_delta,"Delta");
    // print_tuple_rows(edge_2__2_1,"edge_2__2_1");
    int iteration_counter = 0;
    float join_time = 0;
    float merge_time = 0;
    float rebuild_time = 0;
    float flatten_time = 0;
    float set_diff_time = 0;
    while(true) {
        
        // join path delta and edges full
        // TODO: need heuristic for join order
        int reorder_array[2] = {1,3};
        // print_tuple_rows(path_delta, "Path delta before join");
        timer.start_timer();
        binary_join(edge_2__2_1, path_delta, path_newt, reorder_array, 2, grid_size, block_size);
        timer.stop_timer();
        join_time += timer.get_spent_time();
        // print_tuple_rows(path_newt, "Path newt after join "); 
        // std::cout << ">>>>>>>>>>>>>>>" << path_newt->tuples << std::endl;
        if (iteration_counter != 0) {
            // persist current delta into full
            // print_tuple_rows(path_newt, "Path newt before merge");
            float mdt[3] = {0,0};
            // timer.start_timer();
            merge_relation_container(path_full, path_delta, grid_size, block_size, mdt);
            // print_tuple_rows(path_newt, "Path newt after merge "); 
            checkCuda(cudaDeviceSynchronize());
            // timer.stop_timer();
            merge_time += mdt[0];
            flatten_time += mdt[1];
            rebuild_time += mdt[2];
        }
        free_relation(path_delta);

        if (path_newt->tuple_counts == 0) {
            // fixpoint
            break;
        }
                
        // checkCuda(cudaDeviceSynchronize());
        // print_tuple_rows(path_newt, "Path newt before dedup ");

        timer.start_timer();
        tuple_type* deduplicated_newt_tuples;
        checkCuda(cudaMalloc((void**) &deduplicated_newt_tuples, path_newt->tuple_counts*sizeof(tuple_type)));
        tuple_type* deuplicated_end = thrust::set_difference(
            thrust::device,
            path_newt->tuples, path_newt->tuples + path_newt->tuple_counts,
            path_full->tuples, path_full->tuples + path_full->tuple_counts,
            deduplicated_newt_tuples,
            tuple_indexed_less(path_full->index_column_size, path_full->arity));
        u64 deduplicate_size = deuplicated_end - deduplicated_newt_tuples;
        // print_tuple_rows(path_newt, "Path newt after dedup");
        if (deduplicate_size == 0) {
            // fixpoint
            break;
        }
        checkCuda(cudaDeviceSynchronize());
        timer.stop_timer();
        set_diff_time += timer.get_spent_time();

        column_type* deduplicated_raw;
        checkCuda(cudaMalloc((void**) &deduplicated_raw, deduplicate_size*path_newt->arity*sizeof(column_type)));
        flatten_tuples_raw_data<<<grid_size, block_size>>>(deduplicated_newt_tuples, deduplicated_raw,
                                                   deduplicate_size, path_newt->arity);
        // free_relation(path_newt);
    
        // move newt to delta
        load_relation(path_delta, path_full->arity, deduplicated_raw, deduplicate_size,
                      path_full->index_column_size, path_full->index_map_load_factor,
                      grid_size, block_size);
        // print_tuple_rows(path_full, "Path full after load newt");
        
        iteration_counter++;
        // if (iteration_counter == 3) {
        //     break;
        // }
    }
    std::cout << "Finished! path has " << path_full->tuple_counts << std::endl;
    std::cout << "Join time: " << join_time << " ; merge time: " << merge_time
              << " ; rebuild time: " << rebuild_time << " ; flatten time " << flatten_time
              << " ; set diff time: " <<  set_diff_time << std::endl;
    // print_tuple_rows(path_full, "Path full at fix point");
    // reach fixpoint
    
}

//////////////////////////////////////////////////////

long int get_row_size(const char *data_path) {
    std::ifstream f;
    f.open(data_path);
    char c;
    long i = 0;
    while (f.get(c))
        if (c == '\n')
            ++i;
    f.close();
    return  i;
}


column_type *get_relation_from_file(const char *file_path, int total_rows, int total_columns, char separator) {
    column_type *data = (column_type *) malloc(total_rows * total_columns * sizeof(column_type));
    FILE *data_file = fopen(file_path, "r");
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < total_columns; j++) {
            if (j != (total_columns - 1)) {
                fscanf(data_file, "%lld%c", &data[(i * total_columns) + j], &separator);
            } else {
                fscanf(data_file, "%lld", &data[(i * total_columns) + j]);
            }
        }
    }
    return data;
}


void graph_bench(const char* dataset_path, int block_size, int grid_size) {
    KernelTimer timer;
    int relation_columns = 2;
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    time_point_begin = std::chrono::high_resolution_clock::now();
    double spent_time;
    output.initialization_time = 0;
    output.join_time = 0;
    output.memory_clear_time = 0;
    output.total_time = 0;

    // load the raw graph
    u64 graph_edge_counts = get_row_size(dataset_path);
    std::cout << "Input graph rows: " << graph_edge_counts << std::endl;
    // u64 graph_edge_counts = 2100;
    column_type* raw_graph_data = get_relation_from_file(dataset_path, graph_edge_counts, 2, '\t');
    column_type* raw_reverse_graph_data = (column_type *)malloc(graph_edge_counts * 2 * sizeof(column_type));

    std::cout << "reversing graph ... " << std::endl;
    for (u64 i = 0; i < graph_edge_counts; i++) {
        raw_reverse_graph_data[i*2+1] = raw_graph_data[i*2];
        raw_reverse_graph_data[i*2] = raw_graph_data[i*2+1];
    }
    std::cout << "finish reverse graph." << std::endl;

    timer.start_timer();
    GHashRelContainer* edge_2__1_2 = new GHashRelContainer();
    std::cout << "edge size " << graph_edge_counts << std::endl;
    load_relation(edge_2__1_2, 2, raw_graph_data, graph_edge_counts, 1, 0.6, grid_size, block_size);
    GHashRelContainer* edge_2__2_1 = new GHashRelContainer();
    load_relation(edge_2__2_1, 2, raw_reverse_graph_data, graph_edge_counts, 1, 0.6, grid_size, block_size);
    column_type foobar_dummy_data[2] = {0,0};
    GHashRelContainer* result_newt = new GHashRelContainer();
    load_relation(result_newt, 2, foobar_dummy_data, 0, 1, 0.6, grid_size, block_size);
    // checkCuda(cudaDeviceSynchronize());
    timer.stop_timer();
    // double kernel_spent_time = timer.get_spent_time();
    std::cout << "Build hash table time: " << timer.get_spent_time() << std::endl;

    timer.start_timer();
    // edge_2__2_1 ⋈ path_2__1_2
    int reorder_array[2] = {1,3};
    // print_tuple_rows(edge_2__2_1, "edge_2__2_1 before start");
    binary_join(edge_2__2_1, edge_2__1_2, result_newt, reorder_array, 2, grid_size, block_size);
    // print_tuple_rows(result_newt, "Result newt tuples");
    timer.stop_timer();
    std::cout << "join time: " << timer.get_spent_time() << std::endl;
    std::cout << "Result counts: " << result_newt->tuple_counts << std::endl;


    timer.start_timer();
    transitive_closure(edge_2__2_1, edge_2__1_2, block_size, grid_size);
    timer.stop_timer();
    std::cout << "TC time: " << timer.get_spent_time() << std::endl;

}


int main(int argc, char* argv[]) {
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, device_id);
    std::cout << "num of sm " << number_of_sm << std::endl;
    std::cout << "using " << EMPTY_HASH_ENTRY << " as empty hash entry" << std::endl;
    int block_size, grid_size;
    block_size = 512;
    grid_size = 32 * number_of_sm;
    std::locale loc("");

    graph_bench(argv[1], block_size, grid_size);
    return 0;
}
