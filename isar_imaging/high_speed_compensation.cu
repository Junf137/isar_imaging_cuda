#include "high_speed_compensation.cuh"

void highSpeedCompensation(cuComplex* d_data, int Fs, long long band_width, float Tp, float* d_velocity, int echo_num, int range_num, cublasHandle_t handle)
{
	int data_num = echo_num * range_num;

	dim3 block(256);  // block size
	dim3 grid((data_num + block.x - 1) / block.x);  // grid size
	dim3 grid_range((range_num + block.x - 1) / block.x);  // grid size

	// fast time vector
	float* d_tk_2 = nullptr;  // tk_2 = ([0:N-1]/fs).^2
	checkCudaErrors(cudaMalloc((void**)&d_tk_2, sizeof(float)* range_num));
	genTk2Vec << <grid_range, block >> > (d_tk_2, static_cast<float>(Fs), range_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// coef = - 4 * pi * K / c
	float chirp_rate = -static_cast<float>(band_width) / Tp;  // extra minus symbol for velocity (depending on different radar signal)
	float coefficient = 4.0f * PI_h * chirp_rate / lightSpeed_h;

	// phase = coef * v * tk.^2
	float* d_phase = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_phase, sizeof(float) * data_num));  // new mallocated space are set to zero
	checkCudaErrors(cublasSger(handle, range_num, echo_num, &coefficient, d_tk_2, 1, d_velocity, 1, d_phase, range_num));

	// phi = exp(1j*phase)
	cuComplex* d_phi = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_phi, sizeof(cuComplex)* data_num));
	expJ << <grid, block >> > (d_phase, d_phi, data_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// compensation
	elementwiseMultiply << <grid, block >> > (d_data, d_phi, d_data, data_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// free gpu mallocated space
	checkCudaErrors(cudaFree(d_phase));
	checkCudaErrors(cudaFree(d_phi));
	checkCudaErrors(cudaFree(d_tk_2));
}


__global__ void genTk2Vec(float* tk2, float Fs, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		tk2[tid] = (static_cast<float>(tid) / Fs) * (static_cast<float>(tid) / Fs);
	}
}