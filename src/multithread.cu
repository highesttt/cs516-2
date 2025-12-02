// multithread.cu
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

// --- SKELETON ERROR CHECKING ---
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
// --- END SKELETON ---

int* makeRandArray(const int size, const int seed) {
    srand(seed);
    int* array = new int[size];
    for (int i = 0; i < size; i++) {
        array[i] = std::rand() % 1000000;
    }
    return array;
}

// --- KERNEL: Bitonic Sort ---
__global__ void bitonicSortKernel(int* dev_values, int j, int k) {
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i ^ j;

    /* The threads with the lowest ids sort the array. */
    if ((ixj) > i) {
        if ((i & k) == 0) {
            /* Sort ascending */
            if (dev_values[i] > dev_values[ixj]) {
                /* exchange(i, ixj); */
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
        if ((i & k) != 0) {
            /* Sort descending */
            if (dev_values[i] < dev_values[ixj]) {
                /* exchange(i, ixj); */
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int* array;
    int size, seed;
    bool printSorted = false;

    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " [size] [seed] [print 1/0]"
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

    // 1. Calculate Padding (Bitonic sort needs Power of 2)
    int padded_size = 1;
    while (padded_size < size)
        padded_size *= 2;

    int* d_array;
    CudaSafeCall(cudaMalloc(&d_array, padded_size * sizeof(int)));

    // 2. Prepare data with padding (fill extra space with INT_MAX)
    int* host_padded = new int[padded_size];
    for (int i = 0; i < size; i++)
        host_padded[i] = array[i];
    for (int i = size; i < padded_size; i++)
        host_padded[i] = INT_MAX;  // Padding

    CudaSafeCall(cudaMemcpy(d_array, host_padded, padded_size * sizeof(int),
                            cudaMemcpyHostToDevice));

    // 3. Bitonic Sort Launch
    int threadsPerBlock = 256;
    int blocks = (padded_size + threadsPerBlock - 1) / threadsPerBlock;
    int j, k;

    // Major step
    for (k = 2; k <= padded_size; k <<= 1) {
        // Minor step
        for (j = k >> 1; j > 0; j = j >> 1) {
            bitonicSortKernel<<<blocks, threadsPerBlock>>>(d_array, j, k);
            CudaCheckError();
        }
    }

    // 4. Copy back and un-pad
    CudaSafeCall(cudaMemcpy(host_padded, d_array, padded_size * sizeof(int),
                            cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; i++)
        array[i] = host_padded[i];

    cudaFree(d_array);
    delete[] host_padded;

    cudaEventRecord(stopTotal, 0);
    cudaEventSynchronize(stopTotal);
    cudaEventElapsedTime(&timeTotal, startTotal, stopTotal);
    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);

    std::cerr << "Total time in seconds: " << timeTotal / 1000.0 << std::endl;

    if (printSorted) {
        for (int i = 0; i < size; i++)
            cout << array[i] << (i < size - 1 ? ", " : "\n");
    }
    delete[] array;
}