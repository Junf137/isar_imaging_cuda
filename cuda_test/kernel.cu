
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <iomanip>
#include <cublas_v2.h>
#include <cufft.h>
#include <complex>
#include <type_traits>
#include <filesystem>

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
constexpr auto PI_h = 3.14159265358979f;

void cufftTime(int data_num)
{
    // generate data of size echo * range, each element is a float number, pointed by h_data
    std::complex<double>* h_data = new std::complex<double>[data_num];
    for (int i = 0; i < data_num; ++i) {
        h_data[i] = std::complex<double>(static_cast<double>(i + 1), static_cast<double>(i + 1));
    }

    // copy data from host to device
    cuDoubleComplex* d_data = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(cuDoubleComplex) * data_num));
    checkCudaErrors(cudaMemcpy(d_data, h_data, sizeof(cuDoubleComplex) * data_num, cudaMemcpyHostToDevice));

    // create cufft plan
    cufftHandle plan;
    checkCudaErrors(cufftPlan1d(&plan, data_num, CUFFT_Z2Z, 1));

    checkCudaErrors(cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD));

    // execute cufft plan
    int iteration_num = 50;
    auto t_fft_1 = std::chrono::high_resolution_clock::now();
    auto t_fft_2 = std::chrono::high_resolution_clock::now();
    int total_time = 0;

    for (int i = 0; i < iteration_num; ++i) {
        t_fft_1 = std::chrono::high_resolution_clock::now();
        checkCudaErrors(cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD));
        checkCudaErrors(cudaDeviceSynchronize());
        t_fft_2 = std::chrono::high_resolution_clock::now();

        total_time += static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(t_fft_2 - t_fft_1).count());
    }
    total_time /= iteration_num;

    // print total time
    std::cout << "total time: " << total_time << " us" << std::endl;

    // destroy fft handle
    checkCudaErrors(cufftDestroy(plan));

}

__global__ void elementwiseMultiply(cuDoubleComplex* a, cuDoubleComplex* b, cuDoubleComplex* c, int len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        c[tid] = cuCmul(a[tid], b[tid]);
    }
}
void matrixMulTime(int row, int col)
{
    int data_num = row * col;

    dim3 block(1024);
    dim3 grid((data_num + block.x - 1) / block.x);

    // generate data of size data_num, each element is a complex double number, pointed by h_data
    std::complex<double>* h_data = new std::complex<double>[data_num];
    for (int i = 0; i < data_num; ++i) {
        h_data[i] = std::complex<double>(static_cast<double>(i + 1), static_cast<double>(i + 1));
    }

    // copy data from host to device
    cuDoubleComplex* d_data = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(cuDoubleComplex) * data_num));
    checkCudaErrors(cudaMemcpy(d_data, h_data, sizeof(cuDoubleComplex) * data_num, cudaMemcpyHostToDevice));


    int iteration_num = 50;
    auto t_fft_1 = std::chrono::high_resolution_clock::now();
    auto t_fft_2 = std::chrono::high_resolution_clock::now();
    int total_time = 0;

    for (int i = 0; i < iteration_num; ++i) {
        t_fft_1 = std::chrono::high_resolution_clock::now();
        elementwiseMultiply << <grid, block >> > (d_data, d_data, d_data, data_num);
        checkCudaErrors(cudaDeviceSynchronize());
        t_fft_2 = std::chrono::high_resolution_clock::now();

        total_time += static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(t_fft_2 - t_fft_1).count());
    }
    total_time /= iteration_num;

    // print total time
    std::cout << "total time: " << total_time << " us" << std::endl;
}

__global__ void sum(float* vec, float* res, int len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int s = len >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            vec[tid] += vec[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *res = vec[0];
    }
}

__global__ void sumRows(double* d_data, double* d_sum_rows, int rows, int cols)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    double t_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        t_sum = t_sum + d_data[bid * cols + i];
    }

    // Perform a reduction within the block to compute the final sum
    extern __shared__ double sdata_sum_rows_flt[];
    sdata_sum_rows_flt[tid] = t_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_sum_rows_flt[tid] = sdata_sum_rows_flt[tid] + sdata_sum_rows_flt[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_sum_rows[bid] = sdata_sum_rows_flt[0];
    }
}
void sumRowsTime(int rows, int cols)
{
    int data_num = rows * cols;
    double* h_data = new double[data_num];
    for (int i = 0; i < data_num; ++i) {
        h_data[i] = static_cast<double>(i + 1);
    }

    // copy data from host to device
    double* d_data = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(double) * data_num));
    checkCudaErrors(cudaMemcpy(d_data, h_data, sizeof(double) * data_num, cudaMemcpyHostToDevice));

    double* d_res = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_res, sizeof(double) * rows));

    int iteration_num = 50;
    auto t_fft_1 = std::chrono::high_resolution_clock::now();
    auto t_fft_2 = std::chrono::high_resolution_clock::now();
    int total_time = 0;

    for (int i = 0; i < iteration_num; ++i) {
        t_fft_1 = std::chrono::high_resolution_clock::now();
        sumRows << <rows, 1024, sizeof(double) * 1024 >> > (d_data, d_res, rows, cols);
        checkCudaErrors(cudaDeviceSynchronize());
        t_fft_2 = std::chrono::high_resolution_clock::now();

        total_time += static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(t_fft_2 - t_fft_1).count());
    }
    total_time /= iteration_num;

    // print total time
    std::cout << "total time: " << total_time << " us" << std::endl;
}


int main(int argc, char** argv)
{

    // cufftTime
    //cufftTime(256 * 1000);
    //cufftTime(512 * 1000);
    //cufftTime(1024 * 1000);
    //cufftTime(2048 * 1000);
    //cufftTime(4096 * 1000);
    //cufftTime(8192 * 1000);

    // matrixMulTime
    //matrixMulTime(256, 1000);
    //matrixMulTime(512, 1000);
    //matrixMulTime(1024, 1000);
    //matrixMulTime(2048, 1000);
    //matrixMulTime(4096, 1000);
    //matrixMulTime(8192, 1000);


    //sumRowsTime(256, 10000);
    //sumRowsTime(512, 10000);
    //sumRowsTime(1024, 10000);
    //sumRowsTime(2048, 10000);
    //sumRowsTime(4096, 10000);
    //sumRowsTime(8192, 10000);
}
