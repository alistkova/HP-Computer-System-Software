#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t printWithCuda(unsigned int size);
__global__ void messageKernel(unsigned int size);

int main() {
    const unsigned int size = 5;
    cudaError_t cudaStatus;

    if ((cudaStatus = printWithCuda(size)) != cudaSuccess) {
        std::cerr << "CUDA operation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    if ((cudaStatus = cudaDeviceReset()) != cudaSuccess) {
        std::cerr << "Device reset failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    return 0;
}

__global__ void messageKernel(unsigned int size) {
    const unsigned int tid = threadIdx.x;
    if (tid < size) {
        printf("Message from kernel thread %u\n", tid);
    }
}

cudaError_t printWithCuda(unsigned int size) {
    cudaError_t status;

    if ((status = cudaSetDevice(0)) != cudaSuccess) {
        std::cerr << "Device selection error: " << cudaGetErrorString(status) << std::endl;
        return status;
    }

    const dim3 blockDims(size);
    const dim3 gridDims(1);
    
    messageKernel<<<gridDims, blockDims>>>(size);
    
    if ((status = cudaGetLastError()) != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(status) << std::endl;
        return status;
    }

    if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
        std::cerr << "Device sync error: " << cudaGetErrorString(status) << std::endl;
    }

    return status;
}
