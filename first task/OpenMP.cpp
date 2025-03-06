#include <iostream>
#include <omp.h>

int main() {
    #pragma omp parallel 
    {
        int thread_id = omp_get_thread_num();
        int thread_num = omp_get_num_threads();
    
        #pragma omp critical 
        {
            std::cout << "Message from thread " << thread_id << "of" << thread_num << std::endl;
        }
    }
}