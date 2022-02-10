#include <iostream>
#include <cstdio>
#include <complex>
#include <cmath>
#include <iterator>
#include "string.h"
#include <memory>
#include <thrust/complex.h>
#include "cuda_runtime.h"
#include "parameters.h"
#include "device_launch_parameters.h"

using namespace std;

const int samp = SIZE;
const int f_size = 256;

__host__ __device__ unsigned int bitReverse_gpu(unsigned int x, int log2n) {
    int n = 0;
    for (int i = 0; i < log2n; i++) {
        if (x & (1 << i)) n |= 1 << (log2n - 1 - i);
    }
    return n;
}

__host__ __device__ void Hamming_Window_gpu(double window[], int frame_size) {
    const double PI = 3.1415926536;
    for (int i = 0; i < frame_size; i++) window[i] = 0.54f - 0.46f * cos((float)i * 2.0f * PI / (frame_size - 1));
}

template<class Iter_T>
__host__ __device__ void fft_gpu(Iter_T a, Iter_T b, int log2n)
{
    const double PI = 3.1415926536;
    typedef typename iterator_traits<Iter_T>::value_type complex;
    const complex J(0, 1);
    unsigned int n = 1 << log2n;

    for (unsigned int i = 0; i < n; ++i) b[bitReverse_gpu(i, log2n)] = a[i];
    for (int s = 1; s <= log2n; ++s) {
        int m = 1 << s;
        int m2 = m >> 1;
        complex w(1., 0.);
        complex wm = exp(-J * (PI / m2));
        for (int j = 0; j < m2; ++j) {
            for (unsigned int k = j; k < n; k += m) {
                complex t = w * b[k + m2];
                complex u = b[k];
                b[k] = u + t;
                b[k + m2] = u - t;
                w *= wm;
            }
        }
    }
}

__host__ __device__ void process_gpu(int* value, thrust::complex<double>* fft_)
{
    thrust::complex<double> prefetch[samp];
    for (int i = 0; i < samp; i++) {
        prefetch[i] = (double(value[i]) / 2147483648., double(value[2*i+1]) / 2147483648.);
    }

    double window[f_size];
    //hamming window
    for (int i = 0; i < samp; i += f_size / 2) {
        Hamming_Window_gpu(window, f_size);
        for (int j = 0; j < f_size && i + j < samp; j++) {
            prefetch[i + j].imag(prefetch[i + j].imag() * window[j]);
            prefetch[i + j].real(prefetch[i + j].real() * window[j]);
        }
        fft_gpu(prefetch, fft_, 11);
    }
    
}


__global__ void cuda_kernel(int* value, IndexSave* dInd, thrust::complex<double>* fft_)
{
    // complete cuda kernel function
    int i = 0;
    int TotalThread = blockDim.x * gridDim.x;
    int stripe = SIZE / TotalThread;
    int head = (blockIdx.x * blockDim.x + threadIdx.x) * stripe;
    int LoopLim = head + stripe;

    for (i = head; i < LoopLim; i++) {
        process_gpu(value, fft_);
        dInd[i].blockInd_x = blockIdx.x;
        dInd[i].threadInd_x = threadIdx.x;
        dInd[i].head = head;
        dInd[i].stripe = stripe;
    }
};

void GPU_kernel(int* v, IndexSave* indsave, thrust::complex<double>* f_) {

    int* value;
    IndexSave* dInd;
    thrust::complex<double> *fft_;

    // Creat Timing Event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate Memory Space on Device
    cudaMalloc((void**)&value, sizeof(int) * 2*SIZE);
    cudaMalloc((void**)&fft_, sizeof(thrust::complex<double>) * SIZE);

    // Allocate Memory Space on Device (for observation)
    cudaMalloc((void**)&dInd, sizeof(IndexSave) * SIZE);

    // Copy Data to be Calculated
    cudaMemcpy(value, v, sizeof(int) * 2*SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(fft_, f_, sizeof(thrust::complex<double>) * SIZE, cudaMemcpyHostToDevice);

    // Copy Data (indsave array) to device
    cudaMemcpy(dInd, indsave, sizeof(IndexSave) * SIZE, cudaMemcpyHostToDevice);

    // Start Timer
    cudaEventRecord(start, 0);

    // Launch Kernel
    dim3 dimGrid(8);
    dim3 dimBlock(128); 
    cuda_kernel<<<dimGrid, dimBlock>>>(value, dInd, fft_);

    // Stop Timer
    

    // Copy Output back
    cudaMemcpy(v, value, sizeof(int) * 2*SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(f_, fft_, sizeof(thrust::complex<double>) * SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(indsave, dInd, sizeof(IndexSave) * SIZE, cudaMemcpyDeviceToHost);

    // Release Memory Space on Device
    cudaFree(value);
    cudaFree(fft_);
    cudaFree(dInd);

    // Calculate Elapsed Time
    float elapsedTime;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //display GPU Time
    printf("Execution of Method1 is Completed!\n");
    printf("GPU Time: %.4f ms\n", elapsedTime);
    printf(".......................................................\n");
}
