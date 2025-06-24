#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>


double function(double x, double y) {
    return x * x + y * y;
}

int main(int argc, char* argv[]) {
    
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int width = 9000;
    int height = 9000;
    if (argc > 2) {
        width = std::atoi(argv[1]);
        height = std::atoi(argv[2]);
    }

    double* A(nullptr);

    if (world_rank == 0) {
        A = new double[width * height];
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                A[j * width + i] = function(i, j);
            }
        }
    }

    int local_height = height / world_size;
    double *local_A = new double[local_height * width];

    auto start = std::chrono::high_resolution_clock::now();

    MPI_Scatter(A, local_height * width, MPI_DOUBLE, local_A, local_height * width, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double* local_B = new double[local_height * width];
    for (int j = 0; j < local_height; ++j) {
        for (int i = 1; i < width - 1; ++i) {
            local_B[j * width + i] = (local_A[j * width + (i + 1)] - local_A[j * width + (i - 1)]) / 2.0;
        }
    }
    double* B(nullptr);
    if (world_rank == 0) {
        B = new double[height * width];
    }
    MPI_Gather(local_B, local_height * width, MPI_DOUBLE, B, local_height * width, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;

    if (world_rank == 0) {
        std::cout << "Time: " << diff.count() << " milliseconds\n";
    }

    MPI_Finalize();
    return 0;
}
