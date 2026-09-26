#include <iostream>
#include <omp.h>

int main() {
    int total = 0;
    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i < 10000; ++i) {
        total += i;
    }
    std::cout << "OpenMP test total: " << total << ", threads: " << omp_get_max_threads() << std::endl;
    return 0;
}
