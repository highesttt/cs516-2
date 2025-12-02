// thrust.cu
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

// --- SKELETON ERROR CHECKING (REQUIRED) ---
#define CUDA_CHECK_ERROR
#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char* file, const int line) {
#ifdef CUDA_CHECK_ERROR
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
                cudaGetErrorString(err));
        exit(-1);
    }
#endif
    return;
}

inline void __cudaCheckError(const char* file, const int line) {
#ifdef CUDA_CHECK_ERROR
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
                cudaGetErrorString(err));
        exit(-1);
    }
#endif
    return;
}
// --- END SKELETON ERROR CHECKING ---

int* makeRandArray(const int size, const int seed) {
    srand(seed);
    int* array = new int[size];
    for (int i = 0; i < size; i++) {
        array[i] = std::rand() % 1000000;
    }
    return array;
}

// NO KERNEL NEEDED FOR THRUST - LIBRARY HANDLES IT

int main(int argc, char* argv[]) {
    int* array;
    int size, seed;
    bool printSorted = false;

    if (argc < 4) {
        std::cerr << "usage: " << argv[0]
                  << " [amount of random nums to generate] [seed value for "
                     "rand] [1 to print sorted array, 0 otherwise]"
                  << std::endl;
        exit(-1);
    }

    {
        std::stringstream ss1(argv[1]);
        ss1 >> size;
    }
    {
        std::stringstream ss1(argv[2]);
        ss1 >> seed;
    }
    {
        std::stringstream ss1(argv[3]);
        int sortPrint;
        ss1 >> sortPrint;
        if (sortPrint == 1)
            printSorted = true;
    }

    array = makeRandArray(size, seed);

    cudaEvent_t startTotal, stopTotal;
    float timeTotal;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventRecord(startTotal, 0);

    // 1. Transfer to device
    thrust::device_vector<int> d_vec(array, array + size);

    // 2. Sort using Thrust
    thrust::sort(d_vec.begin(), d_vec.end());

    // 3. Transfer back to host (optional for timing, but needed for print)
    if (printSorted) {
        thrust::copy(d_vec.begin(), d_vec.end(), array);
    }

    CudaCheckError();  // Check for errors

    cudaEventRecord(stopTotal, 0);
    cudaEventSynchronize(stopTotal);
    cudaEventElapsedTime(&timeTotal, startTotal, stopTotal);
    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);

    std::cerr << "Total time in seconds: " << timeTotal / 1000.0 << std::endl;

    if (printSorted) {
        for (int i = 0; i < size; i++) {
            cout << array[i] << (i < size - 1 ? ", " : "\n");
        }
    }
    delete[] array;
}