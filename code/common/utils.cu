#include <string>
#include <chrono>
#include <stdlib.h>

using namespace std;
struct Entity {
    int key;
    int value;
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

struct is_equal {
    __host__ __device__
    bool operator()(const Entity &lhs, const Entity &rhs) {
        if ((lhs.key == rhs.key) && (lhs.value == rhs.value))
            return true;
        return false;
    }
};


struct cmp {
    __host__ __device__
    bool operator()(const Entity &lhs, const Entity &rhs) {
        if (lhs.key < rhs.key)
            return true;
        else if (lhs.key > rhs.key)
            return false;
        else {
            if (lhs.value < rhs.value)
                return true;
            else if (lhs.value > rhs.value)
                return false;
            return true;
        }
    }
};

__device__ int get_position(int key, int hash_table_row_size) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key & (hash_table_row_size - 1);
}

void show_time_spent(string message,
                     chrono::high_resolution_clock::time_point time_point_begin,
                     chrono::high_resolution_clock::time_point time_point_end) {
    chrono::duration<double> time_span = time_point_end - time_point_begin;
    cout << message << ": " << time_span.count() << " seconds" << endl;
}

double get_time_spent(string message,
                      chrono::high_resolution_clock::time_point time_point_begin,
                      chrono::high_resolution_clock::time_point time_point_end) {
    chrono::duration<double> time_span = time_point_end - time_point_begin;
    if (message != "")
        cout << message << ": " << time_span.count() << " seconds" << endl;
    return time_span.count();
}

void show_relation(int *data, int total_rows,
                   int total_columns, const char *relation_name,
                   int visible_rows, int skip_zero) {
    int count = 0;
    cout << "Relation name: " << relation_name << endl;
    cout << "===================================" << endl;
    for (int i = 0; i < total_rows; i++) {
        int skip = 0;
        for (int j = 0; j < total_columns; j++) {
            if ((skip_zero == 1) && (data[(i * total_columns) + j] == 0)) {
                skip = 1;
                continue;
            }
            cout << data[(i * total_columns) + j] << " ";
        }
        if (skip == 1)
            continue;
        cout << endl;
        count++;
        if (count == visible_rows) {
            cout << "Result cropped at row " << count << "\n" << endl;
            return;
        }

    }
    cout << "Result counts " << count << "\n" << endl;
    cout << "" << endl;
}

int *get_relation_from_file(const char *file_path, int total_rows, int total_columns, char separator) {
    int *data = (int *) malloc(total_rows * total_columns * sizeof(int));
    FILE *data_file = fopen(file_path, "r");
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < total_columns; j++) {
            if (j != (total_columns - 1)) {
                fscanf(data_file, "%d%c", &data[(i * total_columns) + j], &separator);
            } else {
                fscanf(data_file, "%d", &data[(i * total_columns) + j]);
            }
        }
    }
    return data;
}

void get_relation_from_file_gpu(int *data, const char *file_path, int total_rows, int total_columns, char separator) {
    FILE *data_file = fopen(file_path, "r");
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < total_columns; j++) {
            if (j != (total_columns - 1)) {
                fscanf(data_file, "%d%c", &data[(i * total_columns) + j], &separator);
            } else {
                fscanf(data_file, "%d", &data[(i * total_columns) + j]);
            }
        }
    }
}


void get_random_relation(int *data, int total_rows, int total_columns) {
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < total_columns; j++) {
            data[(i * total_columns) + j] = (rand() % (32767 - 0 + 1)) + 0;
        }
    }
}

void get_string_relation(int *data, int total_rows, int total_columns) {
    int x = 1, y = 2;
    for (int i = 0; i < total_rows; i++) {
        data[(i * total_columns) + 0] = x++;
        data[(i * total_columns) + 1] = y++;
    }
}

void get_reverse_relation_gpu(int *reverse_data, int *data, int total_rows, int total_columns) {
    for (int i = 0; i < total_rows; i++) {
        int pos = total_columns - 1;
        for (int j = 0; j < total_columns; j++) {
            reverse_data[(i * total_columns) + j] = data[(i * total_columns) + pos];
            pos--;
        }
    }
}


void show_hash_table(Entity *hash_table, long int hash_table_row_size, const char *hash_table_name) {
    int count = 0;
    cout << "Hashtable name: " << hash_table_name << endl;
    cout << "===================================" << endl;
    for (int i = 0; i < hash_table_row_size; i++) {
        if (hash_table[i].key != -1) {
            cout << hash_table[i].key << " " << hash_table[i].value << endl;
            count++;
        }
    }
    cout << "Row counts " << count << "\n" << endl;
    cout << "" << endl;
}

void show_entity_array(Entity *data, int data_rows, const char *array_name) {
    long int count = 0;
    cout << "Entity name: " << array_name << endl;
    cout << "===================================" << endl;
    for (int i = 0; i < data_rows; i++) {
        if (data[i].key != -1) {
            cout << data[i].key << " " << data[i].value << endl;
            count++;
        }
    }
    cout << "Row counts " << count << "\n" << endl;
    cout << "" << endl;
}

long int get_row_size(const char *data_path) {
    long int row_size = 0;
    int base = 1;
    for (int i = strlen(data_path) - 1; i >= 0; i--) {
        if (isdigit(data_path[i])) {
            int digit = (int) data_path[i] - '0';
            row_size += base * digit;
            base *= 10;
        }
    }
    return row_size;
}
