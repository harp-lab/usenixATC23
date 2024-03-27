#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
#include <iostream>

using namespace std;

int main() {
    int A1[7] = {0, 2, 4, 6, 8, 10, 12};
    int A2[5] = {2, 3, 5, 7, 9};
    int result[15];
    int *result_end = thrust::set_union(thrust::host, A1, A1 + 7, A2, A2 + 5, result);
    cout << result_end - result << endl;
    std::cout << "Union result: ";
    for (int i = 0; i < result_end - result; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
