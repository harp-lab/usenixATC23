/*
 * Method that returns position in the hashtable for a key using Murmur3 hash
 * */


__global__
void build_hash_table(Entity *hash_table, long int hash_table_row_size,
                      int *relation, long int relation_rows, int relation_columns) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < relation_rows; i += stride) {
        int key = relation[(i * relation_columns) + 0];
        int value = relation[(i * relation_columns) + 1];
        int position = get_position(key, hash_table_row_size);
        while (true) {
            int existing_key = atomicCAS(&hash_table[position].key, -1, key);
            if (existing_key == -1) {
                hash_table[position].value = value;
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
    }
}

__global__
void initialize_result_t_delta(Entity *result, Entity *t_delta,
                       int *relation, long int relation_rows, int relation_columns) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < relation_rows; i += stride) {
        t_delta[i].key = result[i].key = relation[(i * relation_columns) + 0];
        t_delta[i].value = result[i].value = relation[(i * relation_columns) + 1];
    }
}

__global__
void copy_struct(Entity *source, long int source_rows, Entity *destination) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= source_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < source_rows; i += stride) {
        destination[i].key = source[i].key;
        destination[i].value = source[i].value;
    }
}

__global__
void negative_fill_struct(Entity *source, long int source_rows) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= source_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < source_rows; i += stride) {
        source[i].key = -1;
        source[i].value = -1;
    }
}

__global__
void get_reverse_relation(int *relation, long int relation_rows, int relation_columns, Entity *t_delta) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (long int i = index; i < relation_rows; i += stride) {
        t_delta[i].key = relation[(i * relation_columns) + 0];
        t_delta[i].value = relation[(i * relation_columns) + 1];
    }
}


__global__
void get_join_result_size(Entity *hash_table, long int hash_table_row_size,
                          Entity *t_delta, long int relation_rows,
                          int *join_result_size) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < relation_rows; i += stride) {
        int key = t_delta[i].value;
        int current_size = 0;
        int position = get_position(key, hash_table_row_size);
        while (true) {
            if (hash_table[position].key == key) {
                current_size++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
        join_result_size[i] = current_size;
    }
}

__global__
void get_join_result(Entity *hash_table, int hash_table_row_size,
                     Entity *t_delta, int relation_rows, int *offset, Entity *join_result) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < relation_rows; i += stride) {
        int key = t_delta[i].value;
        int value = t_delta[i].key;
        int start_index = offset[i];
        int position = get_position(key, hash_table_row_size);
        while (true) {
            if (hash_table[position].key == key) {
                join_result[start_index].key = value;
                join_result[start_index].value = hash_table[position].value;
                start_index++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
    }
}

__global__
void get_join_result_size_ar(Entity *hash_table, long int hash_table_row_size,
                          int *t_delta, long int relation_rows,
                          int *join_result_size) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < relation_rows; i += stride) {
        int key = t_delta[(i*2) + 1];
        int current_size = 0;
        int position = get_position(key, hash_table_row_size);
        while (true) {
            if (hash_table[position].key == key) {
                current_size++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
        join_result_size[i] = current_size;
    }
}

__global__
void get_join_result_ar(Entity *hash_table, int hash_table_row_size,
                     int *t_delta, int relation_rows, int *offset, Entity *join_result) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < relation_rows; i += stride) {
        int key = t_delta[(i*2) + 1];
        int value = t_delta[i*2];
        int start_index = offset[i];
        int position = get_position(key, hash_table_row_size);
        while (true) {
            if (hash_table[position].key == key) {
                join_result[start_index].key = value;
                join_result[start_index].value = hash_table[position].value;
                start_index++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
    }
}
