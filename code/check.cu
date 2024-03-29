#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <iostream>

using namespace std;
struct Entity {
    int key;
    int value;
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

struct set_cmp {
    __host__ __device__
    bool operator()(const Entity &lhs, const Entity &rhs) {
        if (lhs.key == rhs.key) {
            // If keys are equal, compare values
            return lhs.value < rhs.value;
        }
        return lhs.key < rhs.key;
    }
};


int main() {
    int total_elements;
    // Default data type
    int A1[7] = {0, 2, 4, 6, 8, 10, 12};
    int A2[5] = {2, 3, 5, 7, 9};
    int result[11];
    total_elements = (thrust::set_union(thrust::host,
                                            A1, A1 + 7,
                                            A2, A2 + 5, result)) - result;
    cout << total_elements << endl;
    std::cout << "Union result: ";
    for (int i = 0; i < total_elements; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    // Struct  type
    Entity *a1_entity, *a2_entity, *result_entity;
    a1_entity = (Entity *) malloc(2 * sizeof(Entity));
    a2_entity = (Entity *) malloc(3 * sizeof(Entity));
    result_entity = (Entity *) malloc(5 * sizeof(Entity));

    // a1_entity
    a1_entity[0].key = 1;
    a1_entity[0].value = 3;
    a1_entity[1].key = 2;
    a1_entity[1].value = 1;

    // a2_entity
    a2_entity[0].key = 1;
    a2_entity[0].value = 4;
    a2_entity[1].key = 2;
    a2_entity[1].value = 1;
    a2_entity[2].key = 2;
    a2_entity[2].value = 3;

    // Sort the input arrays
//    thrust::sort(thrust::host, a1_entity, a1_entity + 2, cmp());
//    thrust::sort(thrust::host, a2_entity, a2_entity + 3, cmp());

    total_elements = thrust::set_union(thrust::host,
                                       a1_entity, a1_entity + 2,
                                       a2_entity, a2_entity + 3,
                                       result_entity, set_cmp()) - result_entity;
    cout << total_elements << endl;
    std::cout << "Struct Union result: " << endl;
    for (int i = 0; i < total_elements; ++i) {
        std::cout << result_entity[i].key << " " << result_entity[i].value << endl;
    }
    std::cout << std::endl;

    return 0;
}
