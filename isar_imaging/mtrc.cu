#include "mtrc.cuh"


void mtrc(cuComplex* d_data, const RadarParameters& paras, const CUDAHandle& handles)
{
	dim3 block(DEFAULT_THREAD_PER_BLOCK);
	float scale_ifft_range = 1 / static_cast<float>(paras.range_num_cut);
	float scale_ifft_echo = 1 / static_cast<float>(paras.echo_num);

	float chirp_rate = static_cast<float>(paras.band_width) / static_cast<float>(paras.Tp);
	//posa = K * T_ref / (f0 * (Nr - 1));
	float posa = chirp_rate * static_cast<float>(paras.Tp) / (paras.fc * (paras.range_num_cut - 1.0f));

	// St=ifft(ifftshift(Sf,2),[],2);
	cuComplex* d_st = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_st, sizeof(cuComplex) * paras.data_num_cut));
	checkCudaErrors(cudaMemcpy(d_st, d_data, sizeof(cuComplex) * paras.data_num_cut, cudaMemcpyDeviceToDevice));
	// ifftshift
	ifftshiftRows << <dim3(((paras.range_num_cut / 2) + block.x - 1) / block.x, paras.echo_num), block >> > (d_st, paras.range_num_cut);
	checkCudaErrors(cudaDeviceSynchronize());
	// ifft
	checkCudaErrors(cufftExecC2C(handles.plan_all_echo_c2c_cut, d_st, d_st, CUFFT_INVERSE));
	checkCudaErrors(cublasCsscal(handles.handle, paras.data_num_cut, &scale_ifft_range, d_st, 1));

	// * CZT
	// calculating w and a vector for each range
	cuComplex* d_w = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_w, sizeof(cuComplex) * paras.range_num_cut));
	cuComplex* d_a = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_a, sizeof(cuComplex) * paras.range_num_cut));

	float constant = chirp_rate * static_cast<float>(paras.Tp) / (2 * paras.fc);
	getWandA << <(2 * paras.range_num_cut + block.x - 1) / block.x, block >> > (d_w, d_a, paras.echo_num, paras.range_num_cut, constant, posa);
	checkCudaErrors(cudaDeviceSynchronize());

	cuComplex* d_czt = d_data;
	cztRange(d_czt, d_st, d_w, d_a, paras.echo_num, paras.range_num_cut, handles);
	ifftshiftCols << <dim3(paras.range_num_cut, ((paras.echo_num / 2) + block.x - 1) / block.x), block >> > (d_czt, paras.echo_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// ifft
	checkCudaErrors(cufftExecC2C(handles.plan_all_range_c2c, d_czt, d_czt, CUFFT_INVERSE));
	checkCudaErrors(cublasCsscal(handles.handle, paras.data_num_cut, &scale_ifft_echo, d_czt, 1));
	// ifftshift
	ifftshiftCols << <dim3(paras.range_num_cut, ((paras.echo_num / 2) + block.x - 1) / block.x), block >> > (d_czt, paras.echo_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// fft
	checkCudaErrors(cufftExecC2C(handles.plan_all_echo_c2c_cut, d_czt, d_czt, CUFFT_FORWARD));
	// fftshift
	ifftshiftRows << <dim3(((paras.range_num_cut / 2) + block.x - 1) / block.x, paras.echo_num), block >> > (d_czt, paras.range_num_cut);
	checkCudaErrors(cudaDeviceSynchronize());
	// fftshift
	ifftshiftCols << <dim3(paras.range_num_cut, ((paras.echo_num / 2) + block.x - 1) / block.x), block >> > (d_czt, paras.echo_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// * Free allocated memory
	checkCudaErrors(cudaFree(d_st));
	checkCudaErrors(cudaFree(d_w));
	checkCudaErrors(cudaFree(d_a));
}


__global__ void genWW(cuComplex* d_ww, cuComplex* d_w, int echo_num, int range_num, int ww_len, int fft_len)
{
	int bidx = blockIdx.x;
	int tid = blockDim.x * blockIdx.y + threadIdx.x;

	if (tid < ww_len) {
		int kk = tid - (echo_num - 1);
		float kk2 = kk * kk / 2.0f;
		comThr tmp = thrust::pow(comThr(d_w[bidx].x, d_w[bidx].y), kk2);
		d_ww[tid * range_num + bidx] = make_cuComplex(tmp.real(), tmp.imag());
	}
	else if (tid < fft_len) {
		d_ww[tid * range_num + bidx] = make_cuComplex(0.0f, 0.0f);
	}
}


__global__ void gety(cuComplex* d_y, cuComplex* d_a, cuComplex* d_ww, cuComplex* d_data, int echo_num, int range_num, int y_len, int fft_len)
{
	int bidx = blockIdx.x;
	int tid = blockDim.x * blockIdx.y + threadIdx.x;

	if (tid < y_len) {
		comThr tmp = thrust::pow(comThr(d_a[bidx].x, d_a[bidx].y), static_cast<float>(-tid));
		d_y[tid * range_num + bidx] = cuCmulf(make_cuComplex(tmp.real(), tmp.imag()), d_ww[(echo_num + tid - 1) * range_num + bidx]);
		d_y[tid * range_num + bidx] = cuCmulf(d_y[tid * range_num + bidx], d_data[tid * range_num + bidx]);
	}
	else if (tid < fft_len) {
		d_y[tid * range_num + bidx] = make_cuComplex(0.0f, 0.0f);
	}
}


__global__ void getCZTOut(cuComplex* d_czt, cuComplex* d_ifft, cuComplex* d_ww, int echo_num)
{
	int bidx = blockIdx.x;
	int tid = blockDim.x * blockIdx.y + threadIdx.x;

	int range_num = gridDim.x;

	if (tid < echo_num) {
		d_czt[tid * range_num + bidx] = cuCmulf(d_ifft[(echo_num + tid - 1) * range_num + bidx], d_ww[(echo_num + tid - 1) * range_num + bidx]);
	}
}


void cztRange(cuComplex* d_czt, cuComplex* d_data, cuComplex* d_w, cuComplex* d_a, const int& echo_num, const int& range_num, const CUDAHandle& handles)
{
	dim3 block(DEFAULT_THREAD_PER_BLOCK);

	// nfft = 2^nextpow2(m+k-1);
	int fft_len = nextPow2(2 * echo_num - 1);
	float scale_ifft = 1.0f / fft_len;
	int data_num_fft = fft_len * range_num;
	int ww_len = 2 * echo_num - 1;  // ww length for each range: 2 * echo_num - 1
	int y_len = echo_num;  // y length for each range: echo_num

	cuComplex* d_ww = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_ww, sizeof(cuComplex) * data_num_fft));
	thrust::device_ptr<comThr> thr_ww = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_ww));
	genWW << <dim3(range_num, (fft_len + block.x - 1) / block.x), block >> > (d_ww, d_w, echo_num, range_num, ww_len, fft_len);
	checkCudaErrors(cudaDeviceSynchronize());

	cuComplex* d_y = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_y, sizeof(cuComplex) * data_num_fft));
	gety << <dim3(range_num, (fft_len + block.x - 1) / block.x), block >> > (d_y, d_a, d_ww, d_data, echo_num, range_num, y_len, fft_len);
	checkCudaErrors(cudaDeviceSynchronize());

	// fft
	checkCudaErrors(cufftExecC2C(handles.plan_all_range_c2c_czt, d_y, d_y, CUFFT_FORWARD));

	cuComplex* d_ww_ = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_ww_, sizeof(cuComplex) * data_num_fft));
	thrust::device_ptr<comThr> thr_ww_ = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_ww_));
	thrust::transform(thrust::device, thr_ww, thr_ww + data_num_fft - (fft_len - ww_len) * range_num, thr_ww_, \
		[]__host__ __device__(const comThr & x) { return thrust::pow(x, -1); });

	checkCudaErrors(cufftExecC2C(handles.plan_all_range_c2c_czt, d_ww_, d_ww_, CUFFT_FORWARD));

	elementwiseMultiply << <(data_num_fft + block.x - 1) / block.x, block >> > (d_y, d_ww_, d_y, data_num_fft);
	checkCudaErrors(cudaDeviceSynchronize());

	// ifft
	checkCudaErrors(cufftExecC2C(handles.plan_all_range_c2c_czt, d_y, d_y, CUFFT_INVERSE));
	checkCudaErrors(cublasCsscal(handles.handle, data_num_fft, &scale_ifft, d_y, 1));

	getCZTOut << <dim3(range_num, (echo_num + block.x - 1) / block.x), block >> > (d_czt, d_y, d_ww, echo_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// * Free allocated memory
	checkCudaErrors(cudaFree(d_ww));
	checkCudaErrors(cudaFree(d_y));
	checkCudaErrors(cudaFree(d_ww_));
}


__global__ void getWandA(cuComplex* d_w, cuComplex* d_a, int echo_num, int range_num, float constant, float posa)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < range_num) {
		// calculating w vector
		//w = exp( -1j * 2 * pi * (1 - K * 0.5 * T_ref / f0 + posa * (n - 1)) / Na );
		float tmp = -2 * PI_FLT * (1 - constant + posa * tid) / echo_num;
		d_w[tid] = make_cuComplex(std::cos(tmp), std::sin(tmp));
	}
	else if (tid < 2 * range_num) {
		// calculating a vector
		//a = exp( -1j * pi * (1 - K * 0.5 * T_ref / f0 + posa * (n - 1)) );
		tid -= range_num;
		float tmp = -1 * PI_FLT * (1 - constant + posa * tid);
		d_a[tid] = make_cuComplex(std::cos(tmp), std::sin(tmp));
	}
}
