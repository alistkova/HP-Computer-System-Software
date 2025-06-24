#include <iostream>
#include <chrono>
#include <omp.h>

long long sum_array(size_t arr_size) {
	int* arr = new int[arr_size];

	std::memset(arr, 1, arr_size * sizeof(int)); //fill array with same numbers
	long long sum = 0;

	auto time_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for 
	for (int i = 0; i < arr_size; ++i)
	{
		sum += arr[i];
	}
	
	auto time_end = std::chrono::high_resolution_clock::now();
	delete[] arr;
	std::cout << "Array size, Time elapsed = "  << arr_size;
	std::cout << " " << std::chrono::duration<double, std::milli>(time_end - time_start).count() << "\t milliseconds\n" << std::endl;
	return sum;
}

int main()
{
	const size_t min_size = 1'000'000;
	const size_t max_size = 1'000'000'000;
	omp_set_num_threads(5);

	//log scale
	for (size_t i = min_size; i < max_size; i *= 10)
	{
		for (int j = 1; j < 10; ++j)
		{
			sum_array(i * j);
		}
	}

	return 0;
}
