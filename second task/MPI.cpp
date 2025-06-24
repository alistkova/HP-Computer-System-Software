#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int N = 10000000;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }

    int* data(nullptr);
    if (world_rank == 0) {
        data = new int[N];
        for (int i = 0; i < N; ++i) {
            data[i] = i;
        }
    }

    int local_N = N / world_size;
    int* local_data = new int[local_N];

    auto start = std::chrono::high_resolution_clock::now();

    
    MPI_Scatter(data, local_N, MPI_INT, local_data, local_N, MPI_INT, 0, MPI_COMM_WORLD);

    
    int local_sum = 0;
    for (int i = 0; i < local_N; ++i) { local_sum += local_data[i];}

    int global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;

    if (world_rank == 0) {
        std::cout << "Time: " << diff.count() << " milliseconds\n";
    }
    delete[]data, local_data;
    MPI_Finalize();
    return 0;
}
