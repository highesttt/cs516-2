// singlethread.cu
#include <cuda_runtime.h>

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

// --- KERNEL: Single Threaded Bubble Sort ---
__global__ void singleThreadSort(int* data, int n) {
    // Only thread 0 in block 0 runs this code
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (data[j] > data[j + 1]) {
                    int temp = data[j];
                    data[j] = data[j + 1];
                    data[j + 1] = temp;
                }
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

    int* d_array;
    CudaSafeCall(cudaMalloc(&d_array, size * sizeof(int)));
    CudaSafeCall(
        cudaMemcpy(d_array, array, size * sizeof(int), cudaMemcpyHostToDevice));

    // Launch with 1 block, 1 thread
    singleThreadSort<<<1, 1>>>(d_array, size);
    CudaCheckError();

    CudaSafeCall(
        cudaMemcpy(array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_array);

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