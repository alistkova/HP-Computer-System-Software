#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <chrono>

//grid bounds
const double XMIN = -10.0;
const double XMAX = 10.0;
const double YMIN = -10.0;
const double YMAX = 10.0;

double foo(double x, double y)
{
	return x * x * y + x * y * y;
}

double* generate_grid(size_t xcount, size_t ycount, double* arr)
{

	//double** arr = new double* [xcount];

	double x = XMIN;

	double dx = (XMAX - XMIN) / double(xcount - 1);
	double dy = (YMAX - YMIN) / double(ycount - 1);

	for (int i = 0; i < xcount; ++i)
	{
		//arr[i] = new double[ycount];

		double y = YMIN;

		for (int j = 0; j < ycount; ++j)
		{
			arr[i * xcount + j] = foo(x, y);

			y += dy;
		}

		x += dx;
	}

	return arr;
}

__global__ void matKernel(const double* a, double* b, int size, double dy, double dy2) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size * size) {
		int x = i / size;
		int y = i % size;

		if (!y)
		{
			b[x * size + y] = (a[x * size + y + 1] - a[x * size + y]) / dy;
		}
		else if (y == size - 1)
		{
			b[x * size + y] = (a[x * size + y] - a[x * size + y - 1]) / dy;
		}
		else
		{
			b[x * size + y] = (a[x * size + y + 1] - a[x * size + y - 1]) / dy2;
		}
	}
}

// Helper function for using CUDA to add vectors in parallel.
void derivWithCuda(const double* a, double* b, int size) {
	double dy = (YMAX - YMIN) / double(size - 1);
	double dy2 = dy * 2.0;

	double* dev_a = nullptr;
	double* dev_b = nullptr;

	// Allocate GPU buffers for three vectors (two input, one output)
	cudaMalloc(&dev_a, size * size * sizeof(double));
	cudaMalloc(&dev_b, size * size * sizeof(double));

	// Copy input vectors from host memory to GPU buffers.
	cudaMemcpy(dev_a, a, size * size * sizeof(double), cudaMemcpyHostToDevice);

	// Launch a kernel on the GPU with one thread for each element.
	// 2 is number of computational blocks and (size + 1) / 2 is a number of threads in a block
	matKernel << <2, (size + 1) / 2 >> > (dev_a, dev_b, size, dy, dy2);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaDeviceSynchronize();

	// Copy output vector from GPU buffer to host memory.
	cudaMemcpy(b, dev_b, size * size * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
}

void calc_deriv(size_t N, double* A, double* B) {
	generate_grid(N, N, A);
	auto time_start = std::chrono::high_resolution_clock::now();

	derivWithCuda( A, B, N);

	auto time_end = std::chrono::high_resolution_clock::now();
	std::cout << "Size, Time elapsed: | " << N << " x " << N << " | " << std::chrono::duration<double, std::milli>(time_end - time_start).count() << " |\t milliseconds" << std::endl;
}

int main(int argc, char** argv) {
	const size_t min_size = 1'000;
	const size_t max_size = 10'000;
	double* A = new double[max_size * max_size];
	double* B = new double[max_size * max_size];

	std::memset(A, 1, max_size * max_size * sizeof(double));
	std::memset(B, 1, max_size * max_size * sizeof(double));

	//log scale
	for (size_t i = min_size; i < max_size; i *= 10)
	{
		for (int j = 1; j < 10; ++j)
		{
			calc_deriv(i * j, A, B);
		}
	}

	delete[] A;
	delete[] B;
	return 0;
}