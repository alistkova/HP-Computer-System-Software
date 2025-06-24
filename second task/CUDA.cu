
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>

int atomicAdd(int* address, int val);

__global__ void addKernel(int* c, const int* arr,  int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicAdd(c, arr[i]);
    }
}


int sumWithCuda( const int* arr, int size) {
    int* dev_arr = nullptr, *dev_sum = nullptr;
    int sum = 0;

    // Allocate GPU buffer for vector
    cudaMalloc((void**)&dev_arr, size * sizeof(int));
    cudaMalloc((void**)&dev_sum, sizeof(int));

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sum, dev_sum, sizeof(int), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
    // 2 is number of computational blocks and (size + 1) / 2 is a number of threads in a block
    addKernel << <2, (size + 1) / 2 >> > (dev_sum, dev_arr, size);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(&sum, dev_sum, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_arr);
    cudaFree(dev_sum);
    return sum;
}

int sum_array(size_t arr_size, int* arr) {

    int sum = 0;

    auto time_start = std::chrono::high_resolution_clock::now();

    sum = sumWithCuda(arr, arr_size);

    auto time_end = std::chrono::high_resolution_clock::now();
    std::cout << "Array size, Time elapsed = | " << arr_size;
    std::cout << " | " << std::chrono::duration<double, std::milli>(time_end - time_start).count() << " |\t\t milliseconds" << std::endl;
    return sum;
}

int main()
{
    const size_t min_size = 1'000'000;
    const size_t max_size = 1'000'000'000;
    int* arr = new int[max_size];
    std::memset(arr, 1, max_size * sizeof(int)); //fill array with same numbers
    // Add vectors in parallel.
    
    //log scale
    for (size_t i = min_size; i <= max_size; i *= 10)
    {
        sum_array(i, arr);
        if (i != max_size)
            sum_array(i * 5, arr);
    }

    /*if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "printWithCuda failed!");
        return 1;
    }*/
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}