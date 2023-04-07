
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <iomanip>
#include <cublas_v2.h>
#include <cufft.h>
#include <complex>
#include <type_traits>

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

//void cufftTest()
//{
//    // generate data of size echo * range, each element is a float number, pointed by h_data
//    int echo = 4;
//    int range = 6;
//    int data_num = echo * range;
//    float* h_data = new float[echo * range];
//    for (int i = 0; i < data_num; ++i) {
//        h_data[i] = static_cast<float>(i);
//    }
//
//    // print h_data
//    std::cout << "h_data: \n";
//    for (int i = 0; i < data_num; ++i) {
//        if (i % range == 0) {
//            std::cout << "\n";
//        }
//        std::cout << h_data[i] << " ";
//    }
//
//    // copy data from host to device
//    float* d_data = nullptr;
//    checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(float) * data_num));
//    checkCudaErrors(cudaMemcpy(d_data, h_data, sizeof(float) * data_num, cudaMemcpyHostToDevice));
//
//    cuComplex* d_data_c = nullptr;
//    checkCudaErrors(cudaMalloc((void**)&d_data_c, sizeof(cuComplex) * echo * (range / 2 + 1)));
//
//    // create cufft plan
//    cufftHandle plan;
//    checkCudaErrors(cufftPlan1d(&plan, range, CUFFT_R2C, echo));
//    // execute cufft plan
//    checkCudaErrors(cufftExecR2C(plan, d_data, d_data_c));
//    // copy data from device to host
//    std::complex<float>* h_data_c = new std::complex<float>[echo * (range / 2 + 1)];
//    checkCudaErrors(cudaMemcpy(h_data_c, d_data_c, sizeof(std::complex<float>) * echo * (range / 2 + 1), cudaMemcpyDeviceToHost));
//    // print data
//    std::cout << "\n\nh_data_c: \n";
//    for (int i = 0; i < echo * (range / 2 + 1); ++i) {
//        if (i % (range / 2 + 1) == 0) {
//            std::cout << "\n";
//        }
//        std::cout << h_data_c[i] << " ";
//    }
//
//    // ifft
//    // culabs handle
//    cublasHandle_t handle;
//    checkCudaErrors(cublasCreate(&handle));
//
//    // create cufft plan
//    cufftHandle plan2;
//    checkCudaErrors(cufftPlan1d(&plan2, range, CUFFT_C2R, echo));
//    // execute cufft plan
//    checkCudaErrors(cufftExecC2R(plan2, d_data_c, d_data));
//    // d_data divided by range
//    float scal_ifft = 1 / static_cast<float>(range);
//    checkCudaErrors(cublasSscal(handle, data_num, &scal_ifft, d_data, 1));
//    // copy data from device to host
//    checkCudaErrors(cudaMemcpy(h_data, d_data, sizeof(float) * data_num, cudaMemcpyDeviceToHost));
//    // print data
//    std::cout << "\n\nh_data: \n";
//    for (int i = 0; i < data_num; ++i) {
//        if (i % range == 0) {
//            std::cout << "\n";
//        }
//        std::cout << h_data[i] << " ";
//    }
//}


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


int main(int argc, char** argv)
{
    cuComplex* d_data = new cuComplex[3];
    d_data[0] = make_cuComplex(0.0f, 0.0f);
    d_data[1] = make_cuComplex(1.0f, 0.0f);
    d_data[2] = make_cuComplex(2.0f, 0.0f);

    cuComplex* d_data_tmp = new cuComplex[1];

    checkCudaErrors(cudaMemcpy(d_data_tmp, d_data, sizeof(cuComplex) * 1, cudaMemcpyHostToHost));
    std::cout << d_data_tmp->x << " " << d_data_tmp->y << std::endl;

    checkCudaErrors(cudaMemcpy(d_data_tmp, d_data + 1, sizeof(cuComplex) * 1, cudaMemcpyHostToHost));
    std::cout << d_data_tmp->x << " " << d_data_tmp->y << std::endl;

    checkCudaErrors(cudaMemcpy(d_data_tmp, d_data + 2, sizeof(cuComplex) * 1, cudaMemcpyHostToHost));
    std::cout << d_data_tmp->x << " " << d_data_tmp->y << std::endl;

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
