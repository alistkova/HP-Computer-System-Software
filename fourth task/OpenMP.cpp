#include <iostream>
#include <chrono>
#include <omp.h>

const size_t min_size = 100;
const size_t max_size = 1'700;

void generate_matrix(size_t N, double ** arr)
{
	std::srand(std::time(NULL));

	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			arr[i][j] = -1.0 + double(std::rand() % 20000) / 10000.0;
		}
	}

	return;
}

void matrix_mul(size_t N, double** A, double** B, double** C) {
	//generate_matrix(N, A);
	//generate_matrix(N, B);

	int total_size = N * N;

	auto time_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
	for (int i = 0; i < total_size; ++i)
	{
		int row = i / N;
		int col = i % N;

		for (int j = 0; j < N; ++j)
		{
			C[row][col] += A[row][j] * B[j][col];
		}
	}

	auto time_end = std::chrono::high_resolution_clock::now();
	std::cout << "Size, Time elapsed: " << N << " " << std::chrono::duration<double>(time_end - time_start).count() << "\t seconds" << std::endl;
}

int main(int argc, char* argv[])
{
	omp_set_num_threads(5);
	double** A = new double* [max_size];
	double** B = new double* [max_size];
	double** C = new double* [max_size];
	for (size_t j = 0; j < max_size; ++j)
	{
		A[j] = new double[max_size];
		std::memset(A[j], 1, max_size * sizeof(double));
		B[j] = new double[max_size];
		std::memset(B[j], 1, max_size * sizeof(double));
		C[j] = new double[max_size];
		std::memset(C[j], 0, max_size * sizeof(double));
	}

	//log scale
	/*for (size_t i = min_size; i < max_size; i *= 2)
	{
		for (size_t j = 1; j < 2; ++j)
		{
			matrix_mul(i*j, A, B, C);
		}
	}*/

	//lin scale
	for (size_t i = min_size; i <= max_size; i += 100){
			matrix_mul(i, A, B, C);
	}

	for (size_t i = 0; i < max_size; ++i)
	{
		delete[] A[i];
		delete[] B[i];
		delete[] C[i];
	}
	delete[] A;
	delete[] C;
	delete[] B;
	return 0;
}

