#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>

// Функция для инициализации матрицы случайными значениями
void fill_matrix(double*& matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int N = 1500;

    double *A = new double[N * N], * B = new double[N * N], * C = new double[N * N];
    if (world_rank == 0) {
        fill_matrix(A, N);
        fill_matrix(B, N);
    }

    int rows_per_proc = N / world_size;
    double* local_A = new double[(rows_per_proc * N)];
    double* local_C = new double[(rows_per_proc * N)];

    auto start = std::chrono::high_resolution_clock::now();

    
    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE, local_A, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows_per_proc; ++i) {
        for (int j = 0; j < N; ++j) {
            local_C[i * N + j] = 0.0;
            for (int k = 0; k < N; ++k) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }

    MPI_Gather(local_C, rows_per_proc * N, MPI_DOUBLE, C, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;


    if (world_rank == 0) {
        std::cout << "Time: " << diff.count() << " seconds\n";
    }

    MPI_Finalize();
    delete[] A, B, C, local_A, local_C;
    return 0;
}
