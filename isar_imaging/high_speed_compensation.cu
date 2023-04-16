#include "high_speed_compensation.cuh"


void highSpeedCompensation(cuComplex* d_data, double* d_velocity, const RadarParameters& paras, const CUDAHandle& handles)
{
	dim3 block(DEFAULT_THREAD_PER_BLOCK);  // block size
	dim3 grid((paras.data_num + block.x - 1) / block.x);  // grid size
	dim3 grid_one_echo((paras.range_num + block.x - 1) / block.x);  // grid size

	// fast time vector
	double* d_tk_2 = nullptr;  // tk_2 = ([0:N-1]/fs).^2
	checkCudaErrors(cudaMalloc((void**)&d_tk_2, sizeof(double) * paras.range_num));
	genTk2Vec << <grid_one_echo, block >> > (d_tk_2, static_cast<double>(paras.Fs), paras.range_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// coefficient = - 4 * pi * K / c
	double chirp_rate = static_cast<double>(paras.band_width) / paras.Tp;
	double coefficient = -4.0 * PI_DBL * chirp_rate / static_cast<double>(LIGHT_SPEED);  // extra minus symbol for velocity direction(depending on different radar signal)

	// converting d_data to double precision
	cuDoubleComplex* d_data_dbl = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_data_dbl, sizeof(cuDoubleComplex) * paras.data_num));
	cuComplexFLT2DBL << <grid, block >> > (reinterpret_cast<cuFloatComplex*>(d_data), d_data_dbl, paras.data_num);
	checkCudaErrors(cudaDeviceSynchronize());

	//ioOperation::dataWriteBack(std::string(INTERMEDIATE_DIR) + "dataW.dat", d_data_dbl, paras.data_num);
	//ioOperation::dataWriteBack(std::string(INTERMEDIATE_DIR) + "velocity.dat", d_velocity, paras.echo_num);

	// phase = coefficient * v * tk.^2
	double* d_phase = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_phase, sizeof(double) * paras.data_num));  // new allocated space are set to zero
	checkCudaErrors(cublasDger(handles.handle, paras.range_num, paras.echo_num, &coefficient, d_tk_2, 1, d_velocity, 1, d_phase, paras.range_num));

	//ioOperation::dataWriteBack(std::string(INTERMEDIATE_DIR) + "fai.dat", d_phase, paras.data_num);

	// phi = exp(1j*phase)
	cuDoubleComplex* d_fai = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_fai, sizeof(cuDoubleComplex) * paras.data_num));
	expJ << <grid, block >> > (d_phase, d_fai, paras.data_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// compensation
	elementwiseMultiply << <grid, block >> > (d_data_dbl, d_fai, d_data_dbl, paras.data_num);
	checkCudaErrors(cudaDeviceSynchronize());

	//converting d_data back to float precision
	cuComplexDBL2FLT << <grid, block >> > (reinterpret_cast<cuFloatComplex*>(d_data), d_data_dbl, paras.data_num);
	checkCudaErrors(cudaDeviceSynchronize());

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


__global__ void cuComplexFLT2DBL(cuFloatComplex* d_data_flt, cuDoubleComplex* d_data_dbl, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		d_data_dbl[tid] = cuComplexFloatToDouble(d_data_flt[tid]);
	}
}


__global__ void cuComplexDBL2FLT(cuFloatComplex* d_data_flt, cuDoubleComplex* d_data_dbl, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		d_data_flt[tid] = cuComplexDoubleToFloat(d_data_dbl[tid]);
	}
}
