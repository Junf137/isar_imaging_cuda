
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
constexpr auto PI_h = 3.14159265358979f;


void cufftTest();


void cuBlasTest();


void test();


void cufftTest()
{
    // generate data of size echo * range, each element is a float number, pointed by h_data
    int echo = 4;
    int range = 6;
    int data_num = echo * range;
    float* h_data = new float[echo * range];
    for (int i = 0; i < data_num; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // print h_data
    std::cout << "h_data: \n";
    for (int i = 0; i < data_num; ++i) {
        if (i % range == 0) {
            std::cout << "\n";
        }
        std::cout << h_data[i] << " ";
    }

    // copy data from host to device
    float* d_data = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(float) * data_num));
    checkCudaErrors(cudaMemcpy(d_data, h_data, sizeof(float) * data_num, cudaMemcpyHostToDevice));

    cuComplex* d_data_c = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_data_c, sizeof(cuComplex) * echo * (range / 2 + 1)));

    // create cufft plan
    cufftHandle plan;
    checkCudaErrors(cufftPlan1d(&plan, range, CUFFT_R2C, echo));
    // execute cufft plan
    checkCudaErrors(cufftExecR2C(plan, d_data, d_data_c));
    // copy data from device to host
    std::complex<float>* h_data_c = new std::complex<float>[echo * (range / 2 + 1)];
    checkCudaErrors(cudaMemcpy(h_data_c, d_data_c, sizeof(std::complex<float>) * echo * (range / 2 + 1), cudaMemcpyDeviceToHost));
    // print data
    std::cout << "\n\nh_data_c: \n";
    for (int i = 0; i < echo * (range / 2 + 1); ++i) {
        if (i % (range / 2 + 1) == 0) {
            std::cout << "\n";
        }
        std::cout << h_data_c[i] << " ";
    }

    // ifft
    // culabs handle
    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));

    // create cufft plan
    cufftHandle plan2;
    checkCudaErrors(cufftPlan1d(&plan2, range, CUFFT_C2R, echo));
    // execute cufft plan
    checkCudaErrors(cufftExecC2R(plan2, d_data_c, d_data));
    // d_data divided by range
    float scal_ifft = 1 / static_cast<float>(range);
    checkCudaErrors(cublasSscal(handle, data_num, &scal_ifft, d_data, 1));
    // copy data from device to host
    checkCudaErrors(cudaMemcpy(h_data, d_data, sizeof(float) * data_num, cudaMemcpyDeviceToHost));
    // print data
    std::cout << "\n\nh_data: \n";
    for (int i = 0; i < data_num; ++i) {
        if (i % range == 0) {
            std::cout << "\n";
        }
        std::cout << h_data[i] << " ";
    }
}



/// <summary>
/// Getting the max element of every single row in matrix d_data(incorporating abs operation).
/// Each block is responsible for the calculation of a single row.
/// Since d_data is arranged by row-major in device memory and element per row is usually much bigger than number of thread per block.
/// Therefore calculating partial maximum in each thread before performing reduction on shared memory in block-scale is necessary.
/// Kernel configuration requirements:
/// (1) block_number == rows
/// (2) shared_memory_number == thread_per_block == {256, 512, 1024}
/// </summary>
/// <param name="d_data"></param>
/// <param name="d_max_idx_rows"></param>
/// <param name="rows"></param>
/// <param name="cols"></param>
/// <returns></returns>
__global__ void maxRowsIdxABS(float* d_data, float* d_max_rows_idx, int rows, int cols)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int nTPB = blockDim.x;

    // t_max_rows_idx initialized as the first index handled by this thread
    int t_max_rows_idx = tid;
    for (int i = tid; i < cols; i += nTPB) {
        if (fabs(d_data[bid * cols + t_max_rows_idx]) < fabs(d_data[bid * cols + i])) {
            t_max_rows_idx = i;
        }
    }

    // [todo] Possible optimization: only calculate the first rows in each two stride.
    // Perform a reduction within the block to compute the final maximum value.
    // sdata_max_rows_idx store the index of maximum value in each block.
    extern __shared__ int sdata_max_rows_idx[];
    sdata_max_rows_idx[tid] = t_max_rows_idx;
    __syncthreads();

    for (int s = ((cols < nTPB ? cols : nTPB) >> 1); s > 0; s >>= 1) {
        if (tid < s) {
            if (fabs(d_data[bid * cols + sdata_max_rows_idx[tid]]) < fabs(d_data[bid * cols + sdata_max_rows_idx[tid + s]])) {
                sdata_max_rows_idx[tid] = sdata_max_rows_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        //mopt = maxPos + *h_xstar - NN;
        d_max_rows_idx[bid] = sdata_max_rows_idx[0] + binomialFixDevice(d_data + bid * cols, sdata_max_rows_idx[0]) - (static_cast<float>(cols) / 2);
    }
}

/// <summary>
/// Generating frequency moving vector for every two stride.
/// Kernel configuration requirements:
/// (1) block_number == {(range_num + block.x - 1) / block.x, echo / (stride * 2)}
/// (2) shared_memory_number == thread_per_block == {256, 512, 1024}
/// </summary>
/// <param name="d_freq_mov_vec"></param>
/// <param name="d_max_idx"></param>
/// <param name="rows"></param>
/// <param name="cols"></param>
/// <returns></returns>
__global__ void genFreqMovParallel(cuComplex* d_freq_mov_vec, float* d_max_idx, int cols, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * stride * 2;

    if (idx < cols) {
        float val = -2 * PI_h * static_cast<float>(idx) * d_max_idx[row_idx] / static_cast<float>(cols);
        d_freq_mov_vec[row_idx * cols + idx] = make_cuComplex(std::cos(val), std::sin(val));
    }
}


/// <summary>
/// Performing element-wise multiply to the right stride of each two strides with frequency moving vector stored in first row of each two stride.
/// Kernel configuration requirements:
/// (1) block_number == {(range_num + block.x - 1) / block.x, stride, echo / (stride * 2)}
/// (2) shared_memory_number == thread_per_block == {256, 512, 1024}
/// </summary>
/// <param name="d_data"></param>
/// <param name="d_freq_mov_vec"></param>
/// <param name="cols"></param>
/// <param name="stride"></param>
/// <returns></returns>
__global__ void alignWithinStride(cuComplex* d_data, cuComplex* d_freq_mov_vec, int cols, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base_row_idx = blockIdx.z * stride * 2;
    int row_idx = blockIdx.z * stride * 2 + stride + blockIdx.y;

    if (idx < cols) {
        d_data[row_idx * cols + idx] = cuCmulf(d_data[row_idx * cols + idx], d_freq_mov_vec[base_row_idx * cols + idx]);
    }
}


// copy device data back to host and display
void dDataDisp(float* d_data, int rows, int cols)
{
    float* h_data = new float[rows * cols];
    checkCudaErrors(cudaMemcpy(h_data, d_data, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << h_data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    delete[] h_data;
}


void dDataDisp(cuComplex* d_data, int rows, int cols)
{
    std::complex<float>* h_data = new std::complex<float>[rows * cols];
    checkCudaErrors(cudaMemcpy(h_data, d_data, sizeof(std::complex<float>) * rows * cols, cudaMemcpyDeviceToHost));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << h_data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    delete[] h_data;
}



int main()
{
    int echo_num = 4;
    int range_num = 8;
    int data_num = echo_num * range_num;
    dim3 block(256);

    float* h_data = new float[data_num];
    for (int i = 0; i < data_num; ++i) {
        h_data[i] = static_cast<float>(i);
    }
    // copy data from host to device
    float* d_data = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(float) * data_num));
    checkCudaErrors(cudaMemcpy(d_data, h_data, sizeof(float) * data_num, cudaMemcpyHostToDevice));
    // print d_data
    std::cout << "d_data:" << std::endl;
    dDataDisp(d_data, echo_num, range_num);

    // initializing all element of d_test_data as (1.0f, 0.0f)
    std::complex<float>* h_test_data = new std::complex<float>[data_num];
    for (int i = 0; i < data_num; ++i) {
        h_test_data[i] = std::complex<float>(1.0f, 0.0f);
    }
    cuComplex* d_test_data = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_test_data, sizeof(cuComplex) * data_num));
    checkCudaErrors(cudaMemcpy(d_test_data, h_test_data, sizeof(cuComplex) * data_num, cudaMemcpyHostToDevice));
    // print d_test_data
    std::cout << "d_test_data:" << std::endl;
    dDataDisp(d_test_data, echo_num, range_num);

    // * Initializing memory
    // space for ifft vector and frequency moving vector
    cuComplex* d_com_temp = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_com_temp, sizeof(cuComplex) * data_num));

    // space for average profile
    float* d_ave_profile = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_ave_profile, sizeof(float) * data_num));

    // space for ifft when calculating correlation
    cuComplex* d_ave_profile_fft = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_ave_profile_fft, sizeof(cuComplex) * echo_num * (range_num / 2 + 1)));  // Hermitian symmetry

    // space for storing max value index of every rows
    float* d_max_idx = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_max_idx, sizeof(float) * echo_num));

    // starting align
    int stride = 1;

    // calculating average profile of each stride
    getAveProfileParallel << <dim3(range_num, static_cast<int>(echo_num / stride)), stride, stride * sizeof(float) >> > (d_data, d_ave_profile, echo_num, range_num, stride);
    checkCudaErrors(cudaDeviceSynchronize());
    // print d_ave_profile
    std::cout << "d_ave_profile:" << std::endl;
    dDataDisp(d_ave_profile, echo_num, range_num);

    // cuFFT handle
    cufftHandle plan_all_echo_r2c;
    cufftHandle plan_all_echo_c2r;
    checkCudaErrors(cufftPlan1d(&plan_all_echo_r2c, range_num, CUFFT_R2C, echo_num));
    checkCudaErrors(cufftPlan1d(&plan_all_echo_c2r, range_num, CUFFT_C2R, echo_num));

    // calculating correlation of each two stride's average profile
    // fft
    checkCudaErrors(cufftExecR2C(plan_all_echo_r2c, d_ave_profile, d_ave_profile_fft));
    // print d_ave_profile_fft
    std::cout << "d_ave_profile_fft:" << std::endl;
    dDataDisp(d_ave_profile_fft, echo_num, range_num / 2 + 1);

    // conjugate multiply
    conjMulAveProfile << <range_num, echo_num / (stride * 2) >> > (d_ave_profile_fft, echo_num, range_num / 2 + 1, stride);
    checkCudaErrors(cudaDeviceSynchronize());
    // print d_ave_profile_fft
    std::cout << "d_ave_profile_fft:" << std::endl;
    dDataDisp(d_ave_profile_fft, echo_num, range_num / 2 + 1);

    // ifft
    checkCudaErrors(cufftExecC2R(plan_all_echo_c2r, d_ave_profile_fft, d_ave_profile));
    // print d_ave_profile
    std::cout << "d_ave_profile:" << std::endl;
    dDataDisp(d_ave_profile, echo_num, range_num);

    // ifftshift in each rows
    ifftshiftRows << <echo_num, 256 >> > (d_ave_profile, range_num);
    checkCudaErrors(cudaDeviceSynchronize());
    // print d_ave_profile
    std::cout << "d_ave_profile:" << std::endl;
    dDataDisp(d_ave_profile, echo_num, range_num);

    // getting maximum position in each correlation vector
    maxRowsIdxABS << <echo_num, block, block.x * sizeof(int) >> > (d_ave_profile, d_max_idx, echo_num, range_num);
    checkCudaErrors(cudaDeviceSynchronize());
    // print d_max_idx
    std::cout << "d_max_idx:" << std::endl;
    dDataDisp(d_max_idx, echo_num, 1);

    // aligning second stride in each two stride
    // generating frequency moving vector
    genFreqMovParallel << < dim3((range_num + block.x - 1) / block.x, echo_num / (stride * 2)), block >> > (d_com_temp, d_max_idx, range_num, stride);
    checkCudaErrors(cudaDeviceSynchronize());
    // print d_com_temp
    std::cout << "d_com_temp:" << std::endl;
    dDataDisp(d_com_temp, echo_num, range_num);

    // align
    alignWithinStride << < dim3((range_num + block.x - 1) / block.x, stride, echo_num / (stride * 2)), block >> > (d_test_data, d_com_temp, range_num, stride);
    checkCudaErrors(cudaDeviceSynchronize());
    // print d_test_data
    std::cout << "d_test_data:" << std::endl;
    dDataDisp(d_test_data, echo_num, range_num);
}

void test()
{
    int echo = 2;
    int range = 20;
    int data_num = echo * range;

    std::complex<float>* h_data = new std::complex<float>[data_num];
    for (int i = 0; i < data_num; ++i) {
        h_data[i] = std::complex<float>(static_cast<float>(i), static_cast<float>(i));
    }

    //for (int i = 0; i < data_num; ++i) {
    //    if (i % range == 0) {
    //        std::cout << "\n";
    //    }
    //    std::cout << h_data[i] << " ";
    //}

    cuComplex* d_data = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(cuComplex) * data_num));
    checkCudaErrors(cudaMemcpy(d_data, h_data, sizeof(cuComplex) * data_num, cudaMemcpyHostToDevice));

    cuComplex* d_sum_row = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_sum_row, sizeof(cuComplex) * echo));

    //

    std::complex<float>* h_sum_row = new std::complex<float>[echo];
    checkCudaErrors(cudaMemcpy(h_sum_row, d_sum_row, sizeof(cuComplex) * echo, cudaMemcpyDeviceToHost));

    for (int i = 0; i < echo; ++i) {
        std::cout << h_sum_row[i] << " ";
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