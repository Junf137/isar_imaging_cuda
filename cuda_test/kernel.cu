
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <iomanip>
#include <cublas_v2.h>
#include <cufft.h>
#include <complex>

#include <cuComplex.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/complex.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>


#include "helper_cuda.h"
#include "helper_string.h"

typedef thrust::complex<float> comThr;

void cufftTest();

void cuBlasTest();

void test();

template <typename T>
__global__ void swap_range(T* a, T* b, int len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < len) {
        T c = a[tid]; a[tid] = b[tid]; b[tid] = c;
    }
}


int main()
{
    test();
    //cuBlasTest();
    //cufftTest();

    return 0;
}

void test()
{
    int len = 10;
    float* h_arr = new float[len];
    for (int i = 0; i < len; ++i) {
        h_arr[i] = static_cast<float>(i + 1);
    }

    float* d_arr = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_arr, sizeof(float) * len));
    checkCudaErrors(cudaMemcpy(d_arr, h_arr, sizeof(float) * len, cudaMemcpyHostToDevice));

    swap_range<float> << <1, len >> > (d_arr, d_arr + len / 2, len / 2);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_arr, d_arr, sizeof(float) * len, cudaMemcpyDeviceToHost));
    for (int i = 0; i < len; ++i) {
        std::cout << h_arr[i] << " ";
    }
}

void cuBlasTest()
{
    cublasHandle_t handle;
    cublasCreate(&handle);


    int len = 5;
    float* h_arr = new float[len];
    //for (int i = 0; i < len; ++i) {
    //    h_arr[i] = static_cast<float>(i + 1);
    //}
    h_arr[0] = static_cast<float>(0);
    h_arr[1] = static_cast<float>(1);
    h_arr[2] = static_cast<float>(2);
    h_arr[3] = static_cast<float>(6);
    h_arr[4] = static_cast<float>(3);


    float* d_arr = nullptr;

    float alpha = 5;
    int max_idx = 0;

    checkCudaErrors(cudaMalloc((void**)&d_arr, sizeof(float) * len));
    checkCudaErrors(cudaMemcpy(d_arr, h_arr, sizeof(float) * len, cudaMemcpyHostToDevice));

    //checkCudaErrors(cublasSscal_v2(handle, len, &alpha, d_arr, 1));

    checkCudaErrors(cublasIsamax(handle, len, d_arr, 1, &max_idx));
    --max_idx;
    
    //checkCudaErrors(cudaMemcpy(h_arr, d_arr, sizeof(float) * len, cudaMemcpyDeviceToHost));
    //for (int i = 0; i < len; ++i) {
    //    std::cout << h_arr[i] << " ";
    //}

    std::cout << max_idx;
    
}

void cufftTest() {
    int len = 256 * 4020;
    std::vector<std::complex<float>> h_data(len);

    for (int i = 0; i < len; ++i) {
        h_data[i].real(static_cast<float>(i + 1));
        h_data[i].imag(static_cast<float>(i + 2));
    }

    cuComplex* d_data = nullptr;

    cudaMalloc((void**)&d_data, sizeof(cuComplex) * len);
    cudaMemcpy(d_data, h_data.data(), sizeof(cuComplex) * len, cudaMemcpyHostToDevice);
    thrust::device_ptr<comThr> thr_d_data = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_data));


    //for (int i = 0; i < len; ++i) {
    //    std::cout << thr_d_data[i] << " ";
    //} std::cout << "\n";

    
    cufftHandle plan;
    cufftPlan1d(&plan, len, CUFFT_C2C, 1);

    auto tS = std::chrono::high_resolution_clock::now();
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    auto tE = std::chrono::high_resolution_clock::now();
    std::cout << "[cuFFT] " << std::chrono::duration_cast<std::chrono::milliseconds>(tE - tS).count() << "ms\n";


    //for (int i = 0; i < len; ++i) {
    //    std::cout << thr_d_data[i] << " ";
    //} std::cout << "\n";

    cufftDestroy(plan);

}
