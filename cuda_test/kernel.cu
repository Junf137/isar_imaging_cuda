
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
    //float flt[4] = { 1.0f, 2.0f, 3.0f, 4.0f };

    //// reinterpret_cast test
    //std::complex<float>* cflt = reinterpret_cast<std::complex<float>*>(flt);
    //std::cout << cflt[0] << std::endl;
    //std::cout << cflt[1] << std::endl;

    // define a array of 10 int16 value and initialize it
    int len = 10;
    int16_t* h_data = new int16_t[len];
    for (int i = 0; i < len; ++i) {
		h_data[i] = i + 1;
	}

    // malloc device memory(array of 5 int32 value)
    //cuComplex* d_data = nullptr;
    //float* d_data_flt = reinterpret_cast<float*>(d_data);
    float* d_data_flt = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_data_flt, sizeof(int32_t) * len));

    // copy data from host to device
    checkCudaErrors(cudaMemcpy(d_data_flt, h_data, sizeof(int16_t) * len, cudaMemcpyHostToDevice));

    // print data on device
    dDataDisp(d_data_flt, len, 1);


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


    // * test for sum
    //int len = 8;
    //float* h_data = new float[len];
    //for (int i = 0; i < len; ++i) {
    //    h_data[i] = static_cast<float>(i + 1);
    //}

    //float* d_data = nullptr;
    //checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(float) * len));
    //checkCudaErrors(cudaMemcpy(d_data, h_data, sizeof(float) * len, cudaMemcpyHostToDevice));

    //float* d_res = nullptr;
    //checkCudaErrors(cudaMalloc((void**)&d_res, sizeof(float) * 1));

    //dim3 block(256);
    //dim3 grid((len + block.x - 1) / block.x);
    //sum << <grid, block >> > (d_data, d_res, len);
    //checkCudaErrors(cudaDeviceSynchronize());

    //float* h_res = new float;
    //checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(float) * 1, cudaMemcpyDeviceToHost));

    //std::cout << *h_res << std::endl;


    //int a = 0;
    //float b = 0;
    //std::cout << sizeof(char) << std::endl;
    //std::cout << sizeof(int) << std::endl;
    //std::cout << sizeof(&a) << std::endl;
    //std::cout << sizeof(&b) << std::endl;
    //std::cout << sizeof(float) << std::endl;
    //std::cout << sizeof(double) << std::endl;

    //cuComplex* d_data = new cuComplex[3];
    //d_data[0] = make_cuComplex(0.0f, 0.0f);
    //d_data[1] = make_cuComplex(1.0f, 0.0f);
    //d_data[2] = make_cuComplex(2.0f, 0.0f);

    //cuComplex* d_data_tmp = new cuComplex[1];

    //checkCudaErrors(cudaMemcpy(d_data_tmp, d_data, sizeof(cuComplex) * 1, cudaMemcpyHostToHost));
    //std::cout << d_data_tmp->x << " " << d_data_tmp->y << std::endl;

    //checkCudaErrors(cudaMemcpy(d_data_tmp, d_data + 1, sizeof(cuComplex) * 1, cudaMemcpyHostToHost));
    //std::cout << d_data_tmp->x << " " << d_data_tmp->y << std::endl;

    //checkCudaErrors(cudaMemcpy(d_data_tmp, d_data + 2, sizeof(cuComplex) * 1, cudaMemcpyHostToHost));
    //std::cout << d_data_tmp->x << " " << d_data_tmp->y << std::endl;

 //   int echo_num = 2;
 //   int range_num = 6;
 //   int data_num = echo_num * range_num;
 //   float scale_ifft_range = 1.0f / range_num;
 //   float scale_ifft_echo = 1.0f / echo_num;
 //   dim3 block(256);

 //   cublasHandle_t handle;
 //   checkCudaErrors(cublasCreate(&handle));

 //   cufftHandle plan_all_echo_c2c_cut;
    //checkCudaErrors(cufftPlan1d(&plan_all_echo_c2c_cut, range_num, CUFFT_C2C, echo_num));

 //   cufftHandle plan_all_range_c2c;
 //   cufftHandle plan_all_range_c2c_czt;
 //   int batch = range_num;
 //   int rank = 1;
 //   int n[] = { echo_num };
 //   int inembed[] = { echo_num };
 //   int onembed[] = { echo_num };
 //   int istride = range_num;
 //   int ostride = range_num;
 //   int idist = 1;
 //   int odist = 1;
 //   checkCudaErrors(cufftPlanMany(&plan_all_range_c2c, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch));

 //   int fft_len = nextPow2(2 * echo_num - 1);
 //   n[0] = fft_len;
 //   inembed[0] = fft_len;
 //   onembed[0] = fft_len;
 //   checkCudaErrors(cufftPlanMany(&plan_all_range_c2c_czt, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch));

 //   // initializing all element of d_data
 //   std::complex<float>* h_data = new std::complex<float>[data_num];
 //   for (int i = 0; i < data_num; ++i) {
 //       h_data[i] = std::complex<float>(static_cast<float>(i + 1), 0.0f);
 //   }
 //   cuComplex* d_data = nullptr;
 //   checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(cuComplex) * data_num));
 //   checkCudaErrors(cudaMemcpy(d_data, h_data, sizeof(cuComplex) * data_num, cudaMemcpyHostToDevice));
 //   cuComplex* d_st = nullptr;
 //   checkCudaErrors(cudaMalloc((void**)&d_st, sizeof(cuComplex) * data_num));
 //   checkCudaErrors(cudaMemcpy(d_st, d_data, sizeof(cuComplex) * data_num, cudaMemcpyDeviceToDevice));
 //   // print d_data
 //   std::cout << "d_data:" << std::endl;
 //   dDataDisp(d_data, echo_num, range_num);

 //   // ifftshift
 //   ifftshiftRows << <dim3(((range_num / 2) + block.x - 1) / block.x, echo_num), block >> > (d_st, range_num);
 //   checkCudaErrors(cudaDeviceSynchronize());
 //   // ifft
 //   checkCudaErrors(cufftExecC2C(plan_all_echo_c2c_cut, d_st, d_st, CUFFT_INVERSE));
 //   checkCudaErrors(cublasCsscal(handle, data_num, &scale_ifft_range, d_st, 1));
 //   // print d_data
 //   std::cout << "d_st:" << std::endl;
 //   dDataDisp(d_st, echo_num, range_num);

 //   // * CZT
 //   // calculating w and a vector for each range
 //   cuComplex* d_w = nullptr;
 //   checkCudaErrors(cudaMalloc((void**)&d_w, sizeof(cuComplex) * range_num));
 //   cuComplex* d_a = nullptr;
 //   checkCudaErrors(cudaMalloc((void**)&d_a, sizeof(cuComplex) * range_num));

 //   float constant = 0.0601f;
 //   float posa = 2.2579e-04f;
 //   getWandA << <(2 * range_num + block.x - 1) / block.x, block >> > (d_w, d_a, echo_num, range_num, constant, posa);
 //   checkCudaErrors(cudaDeviceSynchronize());
 //   std::cout << "d_w:" << std::endl;
 //   dDataDisp(d_w, 1, range_num);
 //   std::cout << "d_a:" << std::endl;
 //   dDataDisp(d_a, 1, range_num);

 //   // CZT
 //   // nfft = 2^nextpow2(m+k-1);
 //   float scale_ifft = 1.0f / fft_len;
 //   int data_num_fft = fft_len * range_num;
 //   int ww_len = 2 * echo_num - 1;  // ww length for each range: 2 * echo_num - 1
 //   int y_len = echo_num;  // y length for each range: echo_num

 //   cuComplex* d_ww = nullptr;
 //   checkCudaErrors(cudaMalloc((void**)&d_ww, sizeof(cuComplex) * data_num_fft));
 //   thrust::device_ptr<comThr> thr_ww = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_ww));
 //   genWW << <dim3(range_num, (fft_len + block.x - 1) / block.x), block >> > (d_ww, d_w, echo_num, range_num, ww_len, fft_len);
 //   checkCudaErrors(cudaDeviceSynchronize());

 //   cuComplex* d_y = nullptr;
 //   checkCudaErrors(cudaMalloc((void**)&d_y, sizeof(cuComplex) * data_num_fft));
 //   gety << <dim3(range_num, (fft_len + block.x - 1) / block.x), block >> > (d_y, d_a, d_ww, d_st, echo_num, range_num, y_len, fft_len);
 //   checkCudaErrors(cudaDeviceSynchronize());

 //   // fft
 //   checkCudaErrors(cufftExecC2C(plan_all_range_c2c_czt, d_y, d_y, CUFFT_FORWARD));

 //   cuComplex* d_ww_ = nullptr;
 //   checkCudaErrors(cudaMalloc((void**)&d_ww_, sizeof(cuComplex) * data_num_fft));
 //   thrust::device_ptr<comThr> thr_ww_ = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_ww_));
 //   thrust::transform(thrust::device, thr_ww, thr_ww + data_num_fft - (fft_len - ww_len) * range_num, thr_ww_, \
 //       []__host__ __device__(const comThr & x) { return thrust::pow(x, -1); });

 //   checkCudaErrors(cufftExecC2C(plan_all_range_c2c_czt, d_ww_, d_ww_, CUFFT_FORWARD));

 //   elementwiseMultiply << <(data_num_fft + block.x - 1) / block.x, block >> > (d_y, d_ww_, d_y, data_num_fft);
 //   checkCudaErrors(cudaDeviceSynchronize());

 //   // ifft
 //   checkCudaErrors(cufftExecC2C(plan_all_range_c2c_czt, d_y, d_y, CUFFT_INVERSE));
 //   checkCudaErrors(cublasCsscal(handle, data_num_fft, &scale_ifft, d_y, 1));
 //   std::cout << "d_y(ifft):" << std::endl;
 //   dDataDisp(d_y, fft_len, range_num);

 //   cuComplex* d_czt = d_st;
 //   getCZTOut << <dim3(range_num, (echo_num + block.x - 1) / block.x), block >> > (d_czt, d_y, d_ww, echo_num);
 //   checkCudaErrors(cudaDeviceSynchronize());
 //   std::cout << "d_czt:" << std::endl;
 //   dDataDisp(d_czt, echo_num, range_num);
}
