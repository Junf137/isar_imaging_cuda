#include "range_alignment.cuh"

void rangeAlignment(cuComplex* d_data, float* hamming_window, RadarParameters paras, cublasHandle_t handle, cufftHandle plan_one_echo_c2c, cufftHandle plan_one_echo_r2c, cufftHandle plan_one_echo_c2r)
{
	int data_num = paras.echo_num * paras.range_num;

	float scal_ifft = 1 / static_cast<float>(paras.range_num);  // scalar parameter used after cuFFT ifft transformation

	// * Kernel Thread Comfiguration
	dim3 block(256);  // block size
	dim3 grid_all_echo((data_num + block.x - 1) / block.x);  // grid size
	dim3 grid_one_echo((paras.range_num + block.x - 1) / block.x);
	
	// * Generate Frequency Centering Vector
	cuComplex* d_freq_centering_vec = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_freq_centering_vec, sizeof(cuComplex) * paras.range_num));
	genFreqCenteringVec << <grid_one_echo, block >> > (hamming_window, d_freq_centering_vec, paras.range_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// * Frequency Centering
	elementwiseMultiplyRep << <grid_all_echo, block >> > (d_freq_centering_vec, d_data, d_data, paras.range_num, data_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// * Handle First Echo Signal
	// d_data(1,:) = ifft(d_data(1,:))
	checkCudaErrors(cufftExecC2C(plan_one_echo_c2c, d_data, d_data, CUFFT_INVERSE));
	checkCudaErrors(cublasCsscal(handle, paras.range_num, &scal_ifft, d_data, 1));
	
	float* d_vec_a = nullptr;// vector a
	checkCudaErrors(cudaMalloc((void**)&d_vec_a, sizeof(float) * paras.range_num));

	float* d_vec_b = nullptr;  // vector b = abs(ifft(d_data(1,:)))
	checkCudaErrors(cudaMalloc((void**)&d_vec_b, sizeof(float) * paras.range_num));
	elementwiseAbs << <grid_one_echo, block >> > (d_data, d_vec_b, paras.range_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// * Handle Other Echo Signal in Loop
	// initialization
	cuComplex* d_ifft_temp = nullptr;  // // store fft result in loop
	checkCudaErrors(cudaMalloc((void**)&d_ifft_temp, sizeof(cuComplex) * paras.range_num));

	float* d_vec_corr = nullptr;  // store correlation result in loop
	checkCudaErrors(cudaMalloc((void**)&d_vec_corr, sizeof(float)* paras.range_num));
	
	cuComplex* d_freq_mov_vec = nullptr;  // store alignment result in loop
	checkCudaErrors(cudaMalloc((void**)&d_freq_mov_vec, sizeof(cuComplex) * paras.range_num));

	int NN = paras.range_num / 2;
	int maxPos = 0;
	float mopt = 0.0f;

	float* h_xstar = new float;
	float* d_xstar = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_xstar, sizeof(float) * 1));

	// * starting processing
	for (int i = 1; i < paras.echo_num; i++) {

		cuComplex* d_data_i = d_data + i * paras.range_num;

		// vec_a = abs(ifft(d_data(i,:)));
		checkCudaErrors(cufftExecC2C(plan_one_echo_c2c, d_data_i, d_ifft_temp, CUFFT_INVERSE));
		checkCudaErrors(cublasCsscal(handle, paras.range_num, &scal_ifft, d_ifft_temp, 1));
		elementwiseAbs << <grid_one_echo, block >> > (d_ifft_temp, d_vec_a, paras.range_num);
		checkCudaErrors(cudaDeviceSynchronize());

		// calculate correlation of vec_a and vec_b
		getCorrelation(d_vec_corr, d_vec_a, d_vec_b, paras.range_num, plan_one_echo_r2c, plan_one_echo_c2r);

		// max position in correlation of vec_corr
		checkCudaErrors(cublasIsamax(handle, paras.range_num, d_vec_corr, 1, &maxPos));
		--maxPos;  // cuBlas using 1-based indexing

		// get precise max position using binominal fitting
		binomialFix << <1, 1 >> > (d_vec_corr, d_xstar, maxPos);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(h_xstar, d_xstar, sizeof(float), cudaMemcpyDeviceToHost));

		mopt = maxPos + *h_xstar - NN;

		// d_freq_mov_vec = exp(-1j * 2 * pi * [0:N-1] * mopt / N)
		genFreqMovVec << <grid_one_echo, block >> > (d_freq_mov_vec, mopt, paras.range_num);
		checkCudaErrors(cudaDeviceSynchronize());

		// d_data(i,:) = d_data(i,:) .* d_freq_mov_vec
		elementwiseMultiply << <grid_one_echo, block >> > (d_data_i, d_freq_mov_vec, d_data_i, paras.range_num);
		checkCudaErrors(cudaDeviceSynchronize());

		// d_data(i,:) = ifft(d_data(i,:))
		checkCudaErrors(cufftExecC2C(plan_one_echo_c2c, d_data_i, d_data_i, CUFFT_INVERSE));
		checkCudaErrors(cublasCsscal(handle, paras.range_num, &scal_ifft, d_data_i, 1));

		// update template vector b using aligned vector
		// d_vec_b = 0.95f * d_vec_b + abs(d_data(i,:))
		updateVecB << <grid_all_echo, block >> > (d_vec_b, d_data_i, paras.range_num);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	// step 6: Free GPU Mallocated Space
	delete h_xstar;
	checkCudaErrors(cudaFree(d_xstar));

	checkCudaErrors(cudaFree(d_freq_centering_vec));
	checkCudaErrors(cudaFree(d_vec_a));
	checkCudaErrors(cudaFree(d_vec_b));
	checkCudaErrors(cudaFree(d_ifft_temp));
	checkCudaErrors(cudaFree(d_vec_corr));
	checkCudaErrors(cudaFree(d_freq_mov_vec));
}


__global__ void genFreqCenteringVec(float* hamming, cuComplex* d_freq_centering_vec, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		d_freq_centering_vec[tid] = make_cuComplex(hamming[tid] * std::cos(PI_h * static_cast<float>(tid)), 0.0f);  // todo: __constant__ PI_h variable ???
	}
}


void getCorrelation(float* d_vec_corr, float* d_vec_a, float* d_vec_b, int len, cufftHandle plan_one_echo_r2c, cufftHandle plan_one_echo_c2r)
{
	dim3 block(256);  // block size
	dim3 grid((len + block.x - 1) / block.x);  // grid size

	// fft_a
	cuComplex* d_fft_a = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_fft_a, sizeof(cuComplex) * (len / 2 + 1)));
	checkCudaErrors(cufftExecR2C(plan_one_echo_r2c, d_vec_a, d_fft_a));

	// fft_b
	cuComplex* d_fft_b = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_fft_b, sizeof(cuComplex) * (len / 2 + 1)));
	checkCudaErrors(cufftExecR2C(plan_one_echo_r2c, d_vec_b, d_fft_b));

	// conj multiply, result store in fft_b
	elementwiseMultiplyConjA << <grid, block >> > (d_fft_a, d_fft_b, d_fft_b, len);
	checkCudaErrors(cudaDeviceSynchronize());

	// corr = ifft( conj(fft(a)) * fft(b) )
	checkCudaErrors(cufftExecC2R(plan_one_echo_c2r, d_fft_b, d_vec_corr));

	// fftshift(corr)
	swap_range<float> << <grid, block >> > (d_vec_corr, d_vec_corr + len / 2, len / 2);
	checkCudaErrors(cudaDeviceSynchronize());

	// free GPU mallocated space
	checkCudaErrors(cudaFree(d_fft_a));
	checkCudaErrors(cudaFree(d_fft_b));
}


__global__ void binomialFix(float* d_vec_corr, float* d_xstar, int maxPos)
{
	float f1 = d_vec_corr[maxPos - 1];
	float f2 = d_vec_corr[maxPos];
	float f3 = d_vec_corr[maxPos + 1];

	float fa = (f1 + f3 - 2 * f2) / 2;
	float fb = (f3 - f1) / 2;

	*d_xstar = -fb / (2 * fa);
}


__global__ void updateVecB(float* d_vec_b, cuComplex* d_data_i, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		d_vec_b[tid] = d_vec_b[tid] * 0.95f + cuCabsf(d_data_i[tid]);
	}
}


__global__ void genFreqMovVec(cuComplex* d_freq_mov_vec, float shit_num, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		float val = -2 * PI_h * static_cast<float>(tid) * shit_num / static_cast<float>(len);
		d_freq_mov_vec[tid] = make_cuComplex(std::cos(val), std::sin(val));
	}
}


void HRRPCenter(cuComplex* d_data, RadarParameters paras, const int inter_length, cublasHandle_t handle, cufftHandle plan_all_echo_c2c)
{
	int data_num = paras.echo_num * paras.range_num;

	dim3 block(256);  // block size
	dim3 grid_all_echo((data_num + block.x - 1) / block.x);  // grid size
	dim3 grid_one_echo((paras.range_num + block.x - 1) / block.x);

	// * Normalized HRRP
	float* d_hrrp = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_hrrp, sizeof(float) * data_num));

	elementwiseAbs << <grid_all_echo, block >> > (d_data, d_hrrp, data_num);  // d_hrrp = abs(d_data)
	checkCudaErrors(cudaDeviceSynchronize());

	int hrrp0_max_idx = 0;
	float hrrp0_max_val = 0.0f;  // max(abs(d_hrrp))
	getMax(handle, d_hrrp, data_num, &hrrp0_max_idx, &hrrp0_max_val);
	hrrp0_max_val = 1 / hrrp0_max_val;
	checkCudaErrors(cublasSscal(handle, data_num, &hrrp0_max_val, d_hrrp, 1));  // d_hrrp = d_hrrp / max(abs(d_hrrp))

	// * HRRP_ARP
	thrust::device_vector<float> thr_ones_echo_num(paras.echo_num, 1.0f);
	float* ones_echo_num = reinterpret_cast<float*>(thrust::raw_pointer_cast(thr_ones_echo_num.data()));

	float alpha = 1.0f / float(paras.echo_num);
	float beta = 0.0;

	thrust::device_vector<float> arp(paras.range_num);
	float* d_arp = thrust::raw_pointer_cast(arp.data());  // d_arp = sum(d_hrrp) / echo_num

	checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, paras.range_num, paras.echo_num, &alpha,
		d_hrrp, paras.range_num, ones_echo_num, 1, &beta, d_arp, 1));

	// * Calculate Noise Threshold
	thrust::device_vector<float> arp1(arp.begin(), arp.end());
	float* d_arp1 = thrust::raw_pointer_cast(arp1.data());
	thrust::stable_sort(thrust::device, arp1.begin(), arp1.end());  // todo

	int arp1_max_idx = 0;
	float arp1_max_val = 0.0f;  // max(abs(arp1))
	getMax(handle, d_arp1, paras.range_num, &arp1_max_idx, &arp1_max_val);
	
	float extra_value = static_cast<float>(inter_length) * arp1_max_val / static_cast<float>(paras.range_num);
	int diff_length = paras.range_num - inter_length;

	thrust::device_vector<float> diff(diff_length);
	float* d_diff = thrust::raw_pointer_cast(diff.data());
	thrust::transform(thrust::device, arp1.begin() + inter_length, arp1.end(), arp1.begin(), diff.begin(), \
		[=]__host__ __device__(const float& x, const float& y) { return std::abs(((x - y - extra_value))); });

	int diff_min_idx = 0;
	float diff_min_val = 0.0f;
	getMin(handle, d_diff, diff_length, &diff_min_idx, &diff_min_val);

	float low_threshold_gray = arp1[diff_min_idx + static_cast<int>(inter_length / 2)];
	
	// indices = find( arp > low_threshold_gray )
	thrust::device_vector<int> indices(paras.range_num);
	int* d_indices = thrust::raw_pointer_cast(indices.data());

	thrust::device_vector<int>::iterator end = thrust::copy_if(thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(int(paras.range_num)),
		arp.begin(),
		indices.begin(),
		thrust::placeholders::_1 > low_threshold_gray);
	int indices_length = static_cast<int>(end - indices.begin());
	indices.resize(indices_length);

	int WL = 8;  // window length
	float* d_ARP_ave = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_ARP_ave, sizeof(float) * indices_length));

	dim3 grid_indices((indices_length + block.x - 1) / block.x);
	GetARPMean << <grid_indices, block >> > (d_ARP_ave, d_indices, d_arp, indices_length, WL, paras);
	checkCudaErrors(cudaDeviceSynchronize());

	thrust::device_ptr<float> ARP_ave(d_ARP_ave);

	// ind = find( APR_ave < low_threshold_gray )
	thrust::device_vector<int>ind(indices_length);
	int* d_ind = thrust::raw_pointer_cast(ind.data());
	thrust::device_vector<int>::iterator end_min = thrust::copy_if(thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(indices_length),
		ARP_ave,
		ind.begin(),
		thrust::placeholders::_1 < low_threshold_gray);
	int ind_length = static_cast<int>(end_min - ind.begin());
	ind.resize(ind_length);

	if (ind_length != indices_length) {
		// indices(ind) = 0
		int set_num_len = nextPow2(ind_length);    // power of 2 closest to ind_length.
		dim3 grid_set_num((set_num_len + block.x - 1) / block.x);
		setNumInArray << <grid_set_num, block >> > (d_indices, d_ind, 0, ind_length);
		checkCudaErrors(cudaDeviceSynchronize());

		int mean_indice = thrust::reduce(thrust::device, indices.begin(), indices.end(), 0, thrust::plus<int>()) / (indices_length - ind_length);

		int shift_num = -(mean_indice - paras.range_num / 2 + 1);  // todo: +1???

		// * circshift(d_data,[0, -shiftnum])
		// fft
		checkCudaErrors(cufftExecC2C(plan_all_echo_c2c, d_data, d_data, CUFFT_FORWARD));

		// d_shift_vec = exp(-1j * 2 * pi * [0:N-1] * shift_num)
		cuComplex* d_shift_vec = nullptr;
		checkCudaErrors(cudaMalloc((void**)&d_shift_vec, sizeof(cuComplex)* paras.range_num));
		genFreqMovVec << <grid_one_echo, block >> > (d_shift_vec, shift_num, paras.range_num);
		checkCudaErrors(cudaDeviceSynchronize());

		// d_data = d_data * repmat(d_shift_vec)
		elementwiseMultiplyRep << <grid_all_echo, block >> > (d_shift_vec, d_data, d_data, paras.range_num, data_num);
		checkCudaErrors(cudaDeviceSynchronize());

		// ifft
		checkCudaErrors(cufftExecC2C(plan_all_echo_c2c, d_data, d_data, CUFFT_INVERSE));
		float scal_ifft = 1 / static_cast<float>(paras.range_num);
		checkCudaErrors(cublasCsscal(handle, data_num, &scal_ifft, d_data, 1));
	}

	// * Free GPU Mallocated Space
	checkCudaErrors(cudaFree(d_hrrp));
	checkCudaErrors(cudaFree(d_ARP_ave));
}


//__global__ void circShift(cuComplex* d_in, cuComplex* d_out, int frag_len, int shift_num, int len)
//{
//	int tid = blockIdx.x * blockDim.x + threadIdx.x;
//	if (tid < len)
//	{
//		//int offset = (tid % frag_len + shift_num) % frag_len;
//		//int base = static_cast<int>(tid / frag_len) * frag_len;
//		d_out[static_cast<int>(tid / frag_len) * frag_len + (tid % frag_len + shift_num) % frag_len] = d_in[tid];
//	}
//}


__global__ void GetARPMean(float* ARP_ave, int* indices, float* arp, int indices_length, int WL, RadarParameters paras)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	float temp_sum = 0;
	if (indices[tid] - WL / 2 >= 0 && indices[tid] + WL / 2 <= paras.range_num - 1) {
		for (int index_ARP = indices[tid] - WL / 2; index_ARP <= indices[tid] + WL / 2; index_ARP++) {
			temp_sum += arp[index_ARP];
		}
		temp_sum /= WL + 1;
	}
	else if (indices[tid] - WL / 2 < 0) {
		for (int index_ARP = 0; index_ARP <= indices[tid] + WL / 2; index_ARP++) {
			temp_sum += arp[index_ARP];
		}
		temp_sum /= WL / 2 + indices[tid] + 1;
	}
	else if (indices[tid] + WL / 2 > paras.range_num - 1) {
		for (int index_ARP = indices[tid] - WL / 2; index_ARP < paras.range_num; index_ARP++) {
			temp_sum += arp[index_ARP];
		}
		temp_sum /= (paras.range_num) - (indices[tid] - WL / 2) - 1;
	}
	ARP_ave[tid] = temp_sum;

}

