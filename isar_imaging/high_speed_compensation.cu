#include "high_speed_compensation.cuh"

void highSpeedCompensation(cuComplex* d_data, float* d_velocity, const RadarParameters& paras, const CUDAHandle& handles)
{
	int data_num = paras.echo_num * paras.range_num;

	dim3 block(256);  // block size
	dim3 grid((data_num + block.x - 1) / block.x);  // grid size
	dim3 grid_one_echo((paras.range_num + block.x - 1) / block.x);  // grid size

	// fast time vector
	float* d_tk_2 = nullptr;  // tk_2 = ([0:N-1]/fs).^2
	checkCudaErrors(cudaMalloc((void**)&d_tk_2, sizeof(float)* paras.range_num));
	genTk2Vec << <grid_one_echo, block >> > (d_tk_2, static_cast<float>(paras.Fs), paras.range_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// coefficient = - 4 * pi * K / c
	float chirp_rate = -static_cast<float>(paras.band_width) / paras.Tp;  // extra minus symbol for velocity (depending on different radar signal)
	float coefficient = 4.0f * PI_h * chirp_rate / LIGHT_SPEED_h;

	// phase = coefficient * v * tk.^2
	float* d_phase = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_phase, sizeof(float) * data_num));  // new allocated space are set to zero
	checkCudaErrors(cublasSger(handles.handle, paras.range_num, paras.echo_num, &coefficient, d_tk_2, 1, d_velocity, 1, d_phase, paras.range_num));

	// phi = exp(1j*phase)
	cuComplex* d_phi = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_phi, sizeof(cuComplex)* data_num));
	expJ << <grid, block >> > (d_phase, d_phi, data_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// compensation
	elementwiseMultiply << <grid, block >> > (d_data, d_phi, d_data, data_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// free gpu allocated space
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