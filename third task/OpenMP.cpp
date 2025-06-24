#include <iostream>
#include <chrono>
#include <omp.h>

//grid bounds
const double XMIN = -10.0;
const double XMAX = 10.0;
const double YMIN = -10.0;
const double YMAX = 10.0;

const size_t min_size = 1'000;
const size_t max_size = 10'000;



double foo(double x, double y)
{
	return x * x * y + x * y * y;
}

double** generate_grid(size_t xcount, size_t ycount, double** arr)
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
			arr[i][j] = foo(x, y);

			y += dy;
		}

		x += dx;
	}

	return arr;
}

void calc_deriv(size_t x_size, size_t y_size, double** A, double** B) {
	generate_grid(x_size, y_size, A);

	/*double** B = new double* [x_size];
	for (int j = 0; j < x_size; ++j)
	{
		B[j] = new double[y_size];
	}*/

	int total_size = x_size * y_size;
	double dy = (YMAX - YMIN) / double(y_size - 1);
	double dy2 = dy * 2.0;

	auto time_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
	for (int i = 0; i < total_size; ++i)
	{
		int x = i / y_size;
		int y = i % y_size;

		if (!y)
		{
			B[x][y] = (A[x][y + 1] - A[x][y]) / dy;
		}
		else if (y == y_size - 1)
		{
			B[x][y] = (A[x][y] - A[x][y - 1]) / dy;
		}
		else
		{
			B[x][y] = (A[x][y + 1] - A[x][y - 1]) / dy2;
		}
	}

	auto time_end = std::chrono::high_resolution_clock::now();

	std::cout << "x_size x y_size | Time elapsed: " <<x_size <<" x "<<y_size <<" | " << std::chrono::duration<double, std::milli>(time_end - time_start).count() << " \tmilliseconds" << std::endl;
}


int main()
{
	double** A = new double* [max_size];
	double** B = new double* [max_size];
	for (int j = 0; j < max_size; ++j)
	{
		A[j] = new double[max_size];
		B[j] = new double[max_size];
	}
	omp_set_num_threads(5);

	//log scale
	for (size_t i = min_size; i < max_size; i *= 10)
	{
		for (int j = 1; j < 10; ++j)
		{
			calc_deriv(i * j, i * j, A, B);
		}
	}
	for (int j = 0; j < max_size; ++j)
	{
		delete[] A[j];
		delete[] B[j];
	}
	delete[] A;
	delete[] B;
	return 0;
}
