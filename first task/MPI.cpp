#include <iostream>
#include <mpi.h>

int main(int argc, char* argv[])
{

	int rank, size;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < size; ++i){
        std::cout << "Message from process: " << rank << ", size " << size << std::endl;
        MPI_Barrier(MPI_COMM_WORLD); //synchronization
    }
	
	MPI_Finalize();
}