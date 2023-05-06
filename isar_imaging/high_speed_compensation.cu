#include "high_speed_compensation.cuh"


void highSpeedCompensation(cuDoubleComplex* d_data, double* d_velocity, const RadarParameters& paras, const CUDAHandle& handles)
{
	dim3 block(DEFAULT_THREAD_PER_BLOCK);  // block size

	// fast time vector
	double* d_tk_2 = nullptr;  // tk_2 = ([0:N-1]/fs).^2
	checkCudaErrors(cudaMalloc((void**)&d_tk_2, sizeof(double) * paras.range_num));
	genTk2Vec << <(paras.range_num + block.x - 1) / block.x, block >> > (d_tk_2, static_cast<double>(paras.Fs), paras.range_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// coefficient = - 4 * pi * K / c
	double chirp_rate = static_cast<double>(paras.band_width) / paras.Tp;
	double coefficient = -4.0 * PI_DBL * chirp_rate / static_cast<double>(LIGHT_SPEED);  // extra minus symbol for velocity direction(depending on different radar signal)

	// phase = coefficient * v * tk.^2
	double* d_phase = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_phase, sizeof(double) * paras.data_num));  // new allocated space are set to zero
	checkCudaErrors(cublasDger(handles.handle, paras.range_num, paras.echo_num, &coefficient, d_tk_2, 1, d_velocity, 1, d_phase, paras.range_num));

	// phi = exp(1j*phase)
	cuDoubleComplex* d_fai = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_fai, sizeof(cuDoubleComplex) * paras.data_num));
	expJ << <(paras.data_num + block.x - 1) / block.x, block >> > (d_phase, d_fai, paras.data_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// compensation
	elementwiseMultiply << <(paras.data_num + block.x - 1) / block.x, block >> > (d_data, d_fai, d_data, paras.data_num);
	checkCudaErrors(cudaDeviceSynchronize());

	//ioOperation::dataWriteBack(std::string(INTERMEDIATE_DIR) + "dataW.dat", d_data, paras.data_num);
	//ioOperation::writeFile(std::string(INTERMEDIATE_DIR) + "coff.dat", &coefficient, 1);
	//ioOperation::dataWriteBack(std::string(INTERMEDIATE_DIR) + "velocity.dat", d_velocity, paras.echo_num);
	//ioOperation::dataWriteBack(std::string(INTERMEDIATE_DIR) + "tk2.dat", d_tk_2, paras.range_num);
	//ioOperation::dataWriteBack(std::string(INTERMEDIATE_DIR) + "fai.dat", d_phase, paras.data_num);

	// free gpu allocated space
	checkCudaErrors(cudaFree(d_phase));
	checkCudaErrors(cudaFree(d_fai));
	checkCudaErrors(cudaFree(d_tk_2));
}


__global__ void genTk2Vec(double* d_tk2, double Fs, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		d_tk2[tid] = (static_cast<double>(tid) / Fs) * (static_cast<double>(tid) / Fs);
	}
}
