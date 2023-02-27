
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

typedef thrust::complex<float> comThr;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

void cuBlasTest();

void originTest();

void cufftTest();

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    //cuComplex d_data = make_cuComplex(1.0f, 1.0f);
    cuComplex d_data{};
    d_data.x = 12.0f;
    d_data.y = 1.0f;

    cuComplex* d_ptr = &d_data;
    thrust::device_ptr<comThr> thr_d_ptr = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_ptr));
    std::cout << *thr_d_ptr;


    //cudaMalloc((void**)&d_data, sizeof(cuComplex) * 2);
    //d_data[0].x = 

    //cufftTest();

    return 0;
}

void cufftTest() {
    int len = 5;
    std::vector<std::complex<float>> h_data(len);
    h_data[0].real(0.0f); h_data[0].imag(0.0f);
    h_data[1].real(1.0f); h_data[1].imag(0.0f);
    h_data[2].real(2.0f); h_data[2].imag(0.0f);
    h_data[3].real(3.0f); h_data[3].imag(0.0f);
    h_data[4].real(4.0f); h_data[4].imag(0.0f);
    //h_data[5].real(0.0f); h_data[5].imag(0.0f);
    //h_data[6].real(1.0f); h_data[6].imag(0.0f);
    //h_data[7].real(2.0f); h_data[7].imag(0.0f);
    //h_data[8].real(3.0f); h_data[8].imag(0.0f);
    //h_data[9].real(4.0f); h_data[9].imag(0.0f);

    cuComplex* d_data = nullptr;

    cudaMalloc((void**)&d_data, sizeof(cuComplex) * len);
    cudaMemcpy(d_data, h_data.data(), sizeof(cuComplex) * len, cudaMemcpyHostToDevice);
    thrust::device_ptr<comThr> thr_d_data = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_data));


    for (int i = 0; i < len; ++i) {
        std::cout << thr_d_data[i] << " ";
        if (i == 4) std::cout << "\n";
    }
    std::cout << "\n";


    cufftHandle plan;
    cufftPlan1d(&plan, 5, CUFFT_C2C, 1);
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    for (int i = 0; i < len; ++i) {
        std::cout << thr_d_data[i] << " ";
        if (i == 4) std::cout << "\n";
    }
    std::cout << "\n";

    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    thrust::transform(thrust::device, thr_d_data, thr_d_data + len, thr_d_data, [=]__host__ __device__(const comThr & x) { return x / static_cast<float>(len); });

    for (int i = 0; i < len; ++i) {
        std::cout << thr_d_data[i] << " ";
        if (i == 4) std::cout << "\n";
    }
    std::cout << "\n";

}


void cuBlasTest() {
    std::vector<float> echo({ 1,2,1 });
    std::vector<float> range({ 1,2,3,4,5 });

    int echo_num = echo.size();
    int range_num = range.size();
    int data_num = range_num * echo_num;

    float* d_echo = nullptr;
    cudaMalloc((void**)&d_echo, sizeof(float) * echo_num);
    cudaMemcpy(d_echo, echo.data(), sizeof(float) * echo_num, cudaMemcpyHostToDevice);  // data (host -> device)


    thrust::device_ptr<float> thr_d_echo(d_echo);
    for (int i = 0; i < echo_num; ++i) {
        std::cout << thr_d_echo[i] << " ";
    }
    std::cout << "\n";

    float* d_range = nullptr;
    cudaMalloc((void**)&d_range, sizeof(float) * range_num);
    cudaMemcpy(d_range, range.data(), sizeof(float) * range_num, cudaMemcpyHostToDevice);  // data (host -> device)

    thrust::device_ptr<float> thr_d_range(d_range);
    for (int i = 0; i < range_num; ++i) {
        std::cout << thr_d_range[i] << " ";
    }
    std::cout << "\n";

    float* res = nullptr;
    cudaMalloc((void**)&res, sizeof(float) * data_num);
    cudaMemset(res, 0.0f, sizeof(float) * data_num);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float a = 1;
    cublasSger(handle, range_num, echo_num, &a, d_range, 1, d_echo, 1, res, range_num);

    thrust::device_ptr<float> thr_d_res(res);
    for (int i = 0; i < data_num; ++i) {
        std::cout << thr_d_res[i] << " ";
    }
    std::cout << "\n";

    cublasDestroy(handle);

    float* h_data = new float[data_num];
    cudaMemcpy(h_data, res, sizeof(float) * data_num, cudaMemcpyDeviceToHost);

    for (int i = 0; i < data_num; ++i) {
        std::cout << h_data[i] << " ";
    }
}


void originTest() {
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return ;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return ;
    }
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
