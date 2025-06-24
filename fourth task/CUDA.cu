#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <chrono>

__global__ void matKernel(double* c, const double* a, const double* b, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size * size) {
		int row = i / size;
		int col = i % size;

		for (int j = 0; j < size; ++j)
		{
			c[row * size + col] += a[row * size + j] * b[j * size + col];
		}
	}
}

// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(double* c, const double* a, const double* b, int size) {
	double* dev_a = nullptr;
	double* dev_b = nullptr;
	double* dev_c = nullptr;

	// Allocate GPU buffers for three vectors (two input, one output)
	cudaMalloc(&dev_c, size * size * sizeof(double));
	cudaMalloc(&dev_a, size * size * sizeof(double));
	cudaMalloc(&dev_b, size * size * sizeof(double));

	// Copy input vectors from host memory to GPU buffers.
	cudaMemcpy(dev_a, a, size * size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size * size * sizeof(double), cudaMemcpyHostToDevice);

	// Launch a kernel on the GPU with one thread for each element.
	// 2 is number of computational blocks and (size + 1) / 2 is a number of threads in a block
	matKernel << <2, (size + 1) / 2 >> > (dev_c, dev_a, dev_b, size);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaDeviceSynchronize();

	// Copy output vector from GPU buffer to host memory.
	cudaMemcpy(c, dev_c, size * size * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
}

void matrix_mul(size_t N, double* A, double* B, double* C) {

	int total_size = N * N;

	auto time_start = std::chrono::high_resolution_clock::now();

	addWithCuda(C, A, B, N);

	auto time_end = std::chrono::high_resolution_clock::now();
	std::cout << "Size, Time elapsed: | " << N << " x " << N << " | " << std::chrono::duration<double>(time_end - time_start).count() << " |\t seconds" << std::endl;
}

int main(int argc, char** argv) {
	const size_t min_size = 100;
	const size_t max_size = 10'000;
	double* A = new double[max_size * max_size];
	double* B = new double[max_size * max_size];
	double* C = new double[max_size * max_size];

	std::memset(A, 1, max_size * max_size * sizeof(double));
	std::memset(B, 1, max_size * max_size * sizeof(double));
	std::memset(C, 0, max_size * max_size * sizeof(double));

	//log scale
	/*for (size_t i = min_size; i < max_size; i *= 10)
	{
		for (size_t j = 1; j < 10; j+=5)
		{
			matrix_mul(i*j, A, B, C);
		}
	}*/

	//lin scale
	/*for (size_t i = min_size; i <= max_size; i += 100){
			matrix_mul(i, A, B, C);
	}*/

	//manual scale
	size_t scale[]{ 100,500,1000,1500,2000,2500};
	for (size_t i : scale) {
		matrix_mul(i, A, B, C);
	}

	delete[] A;
	delete[] C;
	delete[] B;
	return 0;
}