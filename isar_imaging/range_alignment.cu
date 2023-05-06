#include "range_alignment.cuh"


//void rangeAlignmentParallel(cuComplex* d_data, float* hamming_window, const RadarParameters& paras, const CUDAHandle& handles)
//{
//	// * Kernel thread configuration
//	dim3 block(DEFAULT_THREAD_PER_BLOCK);  // block size
//	dim3 grid((paras.data_num + block.x - 1) / block.x);  // grid size
//
//	float scale_ifft = 1 / static_cast<float>(paras.range_num);  // scalar parameter used after cuFFT ifft transformation
//
//	// * Frequency centering
//	cuComplex* d_freq_centering = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_freq_centering, sizeof(cuComplex) * paras.range_num));
//	genFreqCenteringVec << <dim3((paras.range_num + block.x - 1) / block.x), block >> > (hamming_window, d_freq_centering, paras.range_num);
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	elementwiseMultiplyRep << <grid, block >> > (d_freq_centering, d_data, d_data, paras.range_num, paras.data_num);
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	// * Merge alignment process
//	// * Initializing memory
//	// space for ifft vector and frequency moving vector
//	cuComplex* d_com_temp = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_com_temp, sizeof(cuComplex) * paras.data_num));
//	// space for abs after ifft
//	float* d_ifft_abs = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_ifft_abs, sizeof(float) * paras.data_num));
//	// space for average profile
//	float* d_ave_profile = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_ave_profile, sizeof(float) * paras.data_num));
//	// space for ifft when calculating correlation
//	cuComplex* d_ave_profile_fft = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_ave_profile_fft, sizeof(cuComplex) * paras.echo_num * (paras.range_num / 2 + 1)));  // Hermitian symmetry
//	// space for storing max value index of every rows
//	float* d_max_idx = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_max_idx, sizeof(float) * paras.echo_num));
//
//	// * MergeAligning d_data till stride equal to echo_num
//	int stride = 1;
//	while (stride < paras.echo_num) {
//		// getting profile of d_data in time domain
//		// ifft
//		checkCudaErrors(cufftExecC2C(handles.plan_all_echo_c2c, d_data, d_com_temp, CUFFT_INVERSE));
//		checkCudaErrors(cublasCsscal(handles.handle, paras.data_num, &scale_ifft, d_com_temp, 1));
//		// abs
//		elementwiseAbs << <grid, block >> > (d_com_temp, d_ifft_abs, paras.data_num);
//		checkCudaErrors(cudaDeviceSynchronize());
//
//		// calculating average profile of each stride
//		getAveProfileParallel << <dim3(paras.range_num, static_cast<int>(paras.echo_num / stride)), stride, stride * sizeof(float) >> > (d_ifft_abs, d_ave_profile, paras.echo_num, paras.range_num, stride);
//		checkCudaErrors(cudaDeviceSynchronize());
//
//		// calculating correlation of each two stride's average profile
//		// fft
//		checkCudaErrors(cufftExecR2C(handles.plan_all_echo_r2c, d_ave_profile, d_ave_profile_fft));
//		// conjugate multiply
//		conjMulAveProfile << <paras.range_num, paras.echo_num / (stride * 2) >> > (d_ave_profile_fft, paras.echo_num, paras.range_num / 2 + 1, stride);
//		checkCudaErrors(cudaDeviceSynchronize());
//		// ifft
//		checkCudaErrors(cufftExecC2R(handles.plan_all_echo_c2r, d_ave_profile_fft, d_ave_profile));
//		// ifftshift in each rows
//		ifftshiftRows << <dim3(((paras.range_num / 2) + block.x - 1) / block.x, paras.echo_num), block >> > (d_ave_profile, paras.range_num);
//		checkCudaErrors(cudaDeviceSynchronize());
//
//		// getting maximum position in each correlation vector
//		maxRowsIdxABS << <paras.echo_num, block, block.x * sizeof(int) >> > (d_ave_profile, d_max_idx, paras.echo_num, paras.range_num);
//		checkCudaErrors(cudaDeviceSynchronize());
//
//		// aligning the second stride in each two stride
//		// generating frequency moving vector
//		genFreqMovParallel << < dim3((paras.range_num + block.x - 1) / block.x, paras.echo_num / (stride * 2)), block >> > (d_com_temp, d_max_idx, paras.range_num, stride);
//		checkCudaErrors(cudaDeviceSynchronize());
//		// align
//		alignWithinStride << < dim3((paras.range_num + block.x - 1) / block.x, stride, paras.echo_num / (stride * 2)), block >> > (d_data, d_com_temp, paras.range_num, stride);
//
//		// continuing next align process
//		stride *= 2;
//	}
//
//	// * Applying ifft to all echoes of d_data
//	checkCudaErrors(cufftExecC2C(handles.plan_all_echo_c2c, d_data, d_data, CUFFT_INVERSE));
//	checkCudaErrors(cublasCsscal(handles.handle, paras.data_num, &scale_ifft, d_data, 1));
//
//	// * Free allocated memory
//	checkCudaErrors(cudaFree(d_com_temp));
//	checkCudaErrors(cudaFree(d_ifft_abs));
//	checkCudaErrors(cudaFree(d_ave_profile));
//	checkCudaErrors(cudaFree(d_ave_profile_fft));
//	checkCudaErrors(cudaFree(d_max_idx));
//	checkCudaErrors(cudaFree(d_freq_centering));
//}


__global__ void getAveProfileParallel(float* d_data, float* d_ave_profile, int rows, int cols, const int& stride)
{
	int tid = threadIdx.x;
	int bidx = blockIdx.x;
	int bidy = blockIdx.y;

	// Calculating the square of each element in the stride
	extern __shared__ float sdata_getAveProfileParallel_flt[];
	// rowIdx = bidy * blockDim.x + tid, colIdx = bidx
	sdata_getAveProfileParallel_flt[tid] = d_data[(bidy * blockDim.x + tid) * cols + bidx] * d_data[(bidy * blockDim.x + tid) * cols + bidx];
	__syncthreads();

	// Perform a reduction within the block to compute the final result
	for (int s = (blockDim.x >> 1); s > 0; s >>= 1) {
		if (tid < s) {
			sdata_getAveProfileParallel_flt[tid] += sdata_getAveProfileParallel_flt[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		// rowIdx = bidy * blockDim.x, colIdx = bidx
		d_ave_profile[bidy * blockDim.x * cols + bidx] = std::sqrtf(sdata_getAveProfileParallel_flt[0]);
	}
}


__global__ void conjMulAveProfile(cuComplex* d_data, int rows, int cols, int stride)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	// first stride: rowIdx = tid * stride * 2, colIdx = bid
	// second stride: rowIdx = tid * stride * 2 + stride, colIdx = bid
	int idx_1 = (tid * stride * 2) * cols + bid;
	int idx_2 = (tid * stride * 2 + stride) * cols + bid;
	d_data[idx_1] = cuCmulf(d_data[idx_1], cuConjf(d_data[idx_2]));
}


__device__ float binomialFixDevice(float* d_vec_corr, int maxPos)
{
	float f1 = d_vec_corr[maxPos - 1];
	float f2 = d_vec_corr[maxPos];
	float f3 = d_vec_corr[maxPos + 1];

	float fa = (f1 + f3 - 2 * f2) / 2;
	float fb = (f3 - f1) / 2;

	return -fb / (2 * fa);
}


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
	// sdata_maxRowsIdxABS_int store the index of maximum value in each block.
	extern __shared__ int sdata_maxRowsIdxABS_int[];
	sdata_maxRowsIdxABS_int[tid] = t_max_rows_idx;
	__syncthreads();

	for (int s = (MIN(cols, nTPB) >> 1); s > 0; s >>= 1) {
		if (tid < s) {
			if (fabs(d_data[bid * cols + sdata_maxRowsIdxABS_int[tid]]) < fabs(d_data[bid * cols + sdata_maxRowsIdxABS_int[tid + s]])) {
				sdata_maxRowsIdxABS_int[tid] = sdata_maxRowsIdxABS_int[tid + s];
			}
		}
		__syncthreads();
	}

	if (tid == 0) {
		//mopt = maxPos + *h_xstar - NN;
		d_max_rows_idx[bid] = sdata_maxRowsIdxABS_int[0] + binomialFixDevice(d_data + bid * cols, sdata_maxRowsIdxABS_int[0]) - (static_cast<float>(cols) / 2);
	}
}


__global__ void genFreqMovParallel(cuComplex* d_freq_mov_vec, float* d_max_idx, int cols, int stride)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int row_idx = blockIdx.y * stride * 2;

	if (idx < cols) {
		float val = -2 * PI_FLT * static_cast<float>(idx) * d_max_idx[row_idx] / static_cast<float>(cols);
		d_freq_mov_vec[row_idx * cols + idx] = make_cuComplex(std::cos(val), std::sin(val));
	}
}


__global__ void alignWithinStride(cuComplex* d_data, cuComplex* d_freq_mov_vec, int cols, int stride)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int base_row_idx = blockIdx.z * stride * 2;
	int row_idx = blockIdx.z * stride * 2 + stride + blockIdx.y;
	
	if (idx < cols) {
		d_data[row_idx * cols + idx] = cuCmulf(d_data[row_idx * cols + idx], d_freq_mov_vec[base_row_idx * cols + idx]);
	}
}


__global__ void genFreqCenteringVec(float* hamming, cuComplex* d_freq_centering_vec, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		d_freq_centering_vec[tid] = make_cuComplex(hamming[tid] * std::cos(PI_FLT * static_cast<float>(tid)), 0.0f);
	}
}


//void rangeAlignment(cuComplex* d_data, float* hamming_window, const RadarParameters& paras, const CUDAHandle& handles)
//{
//	float scale_ifft = 1 / static_cast<float>(paras.range_num);  // scalar parameter used after cuFFT ifft transformation
//
//	// * Kernel thread configuration
//	dim3 block(DEFAULT_THREAD_PER_BLOCK);  // block size
//	dim3 grid((paras.data_num + block.x - 1) / block.x);  // grid size
//	dim3 grid_one_echo((paras.range_num + block.x - 1) / block.x);
//
//	// * Generate frequency centering vector
//	cuComplex* d_com_temp = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_com_temp, sizeof(cuComplex) * paras.range_num));
//	genFreqCenteringVec << <grid_one_echo, block >> > (hamming_window, d_com_temp, paras.range_num);
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	// * Frequency centering
//	elementwiseMultiplyRep << <grid, block >> > (d_com_temp, d_data, d_data, paras.range_num, paras.data_num);
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	// * Range Alignment in Iteration
//	// * Handle First Echo Signal
//	// d_data(1,:) = ifft(d_data(1,:))
//	checkCudaErrors(cufftExecC2C(handles.plan_one_echo_c2c, d_data, d_data, CUFFT_INVERSE));
//	checkCudaErrors(cublasCsscal(handles.handle, paras.range_num, &scale_ifft, d_data, 1));
//
//	float* d_vec_a = nullptr;// vector a
//	checkCudaErrors(cudaMalloc((void**)&d_vec_a, sizeof(float) * paras.range_num));
//
//	float* d_vec_b = nullptr;  // vector b = abs(d_data(1,:))
//	checkCudaErrors(cudaMalloc((void**)&d_vec_b, sizeof(float) * paras.range_num));
//	elementwiseAbs << <grid_one_echo, block >> > (d_data, d_vec_b, paras.range_num);
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	// * Handle Other Echo Signal in Loop
//	// initialization
//	float* d_vec_corr = nullptr;  // store correlation result in loop
//	checkCudaErrors(cudaMalloc((void**)&d_vec_corr, sizeof(float) * paras.range_num));
//
//	int NN = paras.range_num / 2;
//	int maxPos = 0;
//	float mopt = 0.0f;
//
//	float* h_xstar = new float;
//	float* d_xstar = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_xstar, sizeof(float) * 1));
//
//	// * starting processing
//	for (int i = 1; i < paras.echo_num; i++) {
//
//		cuComplex* d_data_i = d_data + i * paras.range_num;
//
//		// vec_a = abs(ifft(d_data(i,:)));
//		checkCudaErrors(cufftExecC2C(handles.plan_one_echo_c2c, d_data_i, d_com_temp, CUFFT_INVERSE));
//		checkCudaErrors(cublasCsscal(handles.handle, paras.range_num, &scale_ifft, d_com_temp, 1));
//		elementwiseAbs << <grid_one_echo, block >> > (d_com_temp, d_vec_a, paras.range_num);
//		checkCudaErrors(cudaDeviceSynchronize());
//
//		// calculate correlation of vec_a and vec_b
//		getCorrelation(d_vec_corr, d_vec_a, d_vec_b, paras.range_num, handles.plan_one_echo_r2c, handles.plan_one_echo_c2r);
//
//		// max position in correlation of vec_corr
//		checkCudaErrors(cublasIsamax(handles.handle, paras.range_num, d_vec_corr, 1, &maxPos));
//		--maxPos;  // cuBlas using 1-based indexing
//
//		// get precise max position using binomial fitting
//		binomialFix << <1, 1 >> > (d_vec_corr, d_xstar, maxPos);
//		checkCudaErrors(cudaDeviceSynchronize());
//		checkCudaErrors(cudaMemcpy(h_xstar, d_xstar, sizeof(float), cudaMemcpyDeviceToHost));
//
//		mopt = maxPos + *h_xstar - NN;
//
//		// d_freq_mov_vec = exp(-1j * 2 * pi * [0:N-1] * mopt / N)
//		genFreqMovVec << <grid_one_echo, block >> > (d_com_temp, mopt, paras.range_num);
//		checkCudaErrors(cudaDeviceSynchronize());
//
//		// d_data(i,:) = d_data(i,:) .* d_freq_mov_vec
//		elementwiseMultiply << <grid_one_echo, block >> > (d_com_temp, d_data_i, d_data_i, paras.range_num);
//		checkCudaErrors(cudaDeviceSynchronize());
//
//		// d_data(i,:) = ifft(d_data(i,:))
//		checkCudaErrors(cufftExecC2C(handles.plan_one_echo_c2c, d_data_i, d_data_i, CUFFT_INVERSE));
//		checkCudaErrors(cublasCsscal(handles.handle, paras.range_num, &scale_ifft, d_data_i, 1));
//
//		// update template vector b using aligned vector
//		// d_vec_b = 0.95f * d_vec_b + abs(d_data(i,:))
//		updateVecB << <grid, block >> > (d_vec_b, d_data_i, paras.range_num);
//		checkCudaErrors(cudaDeviceSynchronize());
//	}
//
//	// * Free GPU Allocated Space
//	delete h_xstar;
//	checkCudaErrors(cudaFree(d_xstar));
//	checkCudaErrors(cudaFree(d_vec_a));
//	checkCudaErrors(cudaFree(d_vec_b));
//	checkCudaErrors(cudaFree(d_vec_corr));
//	checkCudaErrors(cudaFree(d_com_temp));
//}


//void getCorrelation(float* d_vec_corr, float* d_vec_a, float* d_vec_b, int len, cufftHandle plan_one_echo_r2c, cufftHandle plan_one_echo_c2r)
//{
//	// * configuring data layout 
//	int fft_len = static_cast<int>(len / 2) + 1;
//
//	dim3 block(DEFAULT_THREAD_PER_BLOCK);
//	dim3 grid((len + block.x - 1) / block.x);
//	dim3 grid_fft((fft_len + block.x - 1) / block.x);
//
//	// * fft_a
//	cuComplex* d_fft_a = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_fft_a, sizeof(cuComplex) * fft_len));
//	checkCudaErrors(cufftExecR2C(plan_one_echo_r2c, d_vec_a, d_fft_a));
//
//	// * fft_b
//	cuComplex* d_fft_b = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_fft_b, sizeof(cuComplex) * fft_len));
//	checkCudaErrors(cufftExecR2C(plan_one_echo_r2c, d_vec_b, d_fft_b));
//
//	// * conj multiply, result store in fft_b
//	elementwiseMultiplyConjA << <grid_fft, block >> > (d_fft_a, d_fft_b, d_fft_b, fft_len);
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	// * corr = ifft( conj(fft(a)) * fft(b) )
//	checkCudaErrors(cufftExecC2R(plan_one_echo_c2r, d_fft_b, d_vec_corr));
//
//	// * fftshift(corr)
//	swap_range<float> << <grid, block >> > (d_vec_corr, d_vec_corr + len / 2, len / 2);
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	// * free GPU allocated space
//	checkCudaErrors(cudaFree(d_fft_a));
//	checkCudaErrors(cudaFree(d_fft_b));
//}


//__global__ void binomialFix(float* d_vec_corr, float* d_xstar, int maxPos)
//{
//	float f1 = d_vec_corr[maxPos - 1];
//	float f2 = d_vec_corr[maxPos];
//	float f3 = d_vec_corr[maxPos + 1];
//
//	float fa = (f1 + f3 - 2 * f2) / 2;
//	float fb = (f3 - f1) / 2;
//
//	*d_xstar = -fb / (2 * fa);
//}


//__global__ void updateVecB(float* d_vec_b, cuComplex* d_data_i, int len)
//{
//	int tid = blockIdx.x * blockDim.x + threadIdx.x;
//	if (tid < len) {
//		d_vec_b[tid] = d_vec_b[tid] * 0.95f + cuCabsf(d_data_i[tid]);
//	}
//}


__global__ void genFreqMovVec(cuComplex* d_freq_mov_vec, float shit_num, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		float val = -2 * PI_FLT * static_cast<float>(tid) * shit_num / static_cast<float>(len);
		d_freq_mov_vec[tid] = make_cuComplex(std::cos(val), std::sin(val));
	}
}


//void HRRPCenter(cuComplex* d_data, const int& inter_length, const RadarParameters& paras, const CUDAHandle& handles)
//{
//	dim3 block(DEFAULT_THREAD_PER_BLOCK);  // block size
//	dim3 grid((paras.data_num + block.x - 1) / block.x);  // grid size
//	dim3 grid_one_echo((paras.range_num + block.x - 1) / block.x);
//
//	// * Normalizing HRRP
//	float* d_hrrp = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_hrrp, sizeof(float) * paras.data_num));
//
//	// d_hrrp = abs(d_data)
//	elementwiseAbs << <grid, block >> > (d_data, d_hrrp, paras.data_num);
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	// d_hrrp = d_hrrp / max(abs(d_hrrp))
//	int hrrp_max_idx = 0;
//	float hrrp_max_val = 0.0f;
//	getMax(handles.handle, d_hrrp, paras.data_num, &hrrp_max_idx, &hrrp_max_val);
//	hrrp_max_val = 1 / hrrp_max_val;
//	checkCudaErrors(cublasSscal(handles.handle, paras.data_num, &hrrp_max_val, d_hrrp, 1));
//
//	// * HRRP_ARP
//	thrust::device_vector<float> arp(paras.range_num);
//	float* d_arp = thrust::raw_pointer_cast(arp.data());
//
//	// d_arp = sum(d_hrrp, 1) / echo_num
//	sumCols << <paras.range_num, 256, 256 * sizeof(float) >> > (d_hrrp, d_arp, paras.echo_num, paras.range_num);
//	checkCudaErrors(cudaDeviceSynchronize());
//	float alpha = 1.0f / static_cast<float>(paras.echo_num);
//	checkCudaErrors(cublasSscal(handles.handle, paras.range_num, &alpha, d_arp, 1));
//
//	// * Calculate Noise Threshold
//	thrust::device_vector<float> arp1(arp.begin(), arp.end());
//	float* d_arp1 = thrust::raw_pointer_cast(arp1.data());
//	thrust::stable_sort(thrust::device, arp1.begin(), arp1.end());
//
//	int arp1_max_idx = 0;
//	float arp1_max_val = 0.0f;  // max(abs(arp1))
//	getMax(handles.handle, d_arp1, paras.range_num, &arp1_max_idx, &arp1_max_val);
//
//	float extra_value = static_cast<float>(inter_length) * arp1_max_val / static_cast<float>(paras.range_num);
//	int diff_length = paras.range_num - inter_length;
//
//	thrust::device_vector<float> diff(diff_length);
//	float* d_diff = thrust::raw_pointer_cast(diff.data());
//	thrust::transform(thrust::device, arp1.begin() + inter_length, arp1.end(), arp1.begin(), diff.begin(), \
//		[=]__host__ __device__(const float& x, const float& y) { return std::abs(x - y - extra_value); });
//
//	int diff_min_idx = 0;
//	float diff_min_val = 0.0f;
//	getMin(handles.handle, d_diff, diff_length, &diff_min_idx, &diff_min_val);
//
//	int low_threshold_gray_idx = diff_min_idx + static_cast<int>(inter_length / 2);
//	float low_threshold_gray = arp1[low_threshold_gray_idx];
//
//	// idx_1 = find( arp > low_threshold_gray )
//	thrust::device_vector<int> idx_1(paras.range_num);
//
//	auto end_idx_1 = thrust::copy_if(thrust::make_counting_iterator(0), thrust::make_counting_iterator(int(paras.range_num)), \
//		arp.begin(), idx_1.begin(), thrust::placeholders::_1 > low_threshold_gray);
//	int idx_1_len = static_cast<int>(end_idx_1 - idx_1.begin());
//	idx_1.resize(idx_1_len);
//	int* d_idx_1 = thrust::raw_pointer_cast(idx_1.data());
//
//	int WL = 8;  // window length
//	float* d_arp_ave = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_arp_ave, sizeof(float) * idx_1_len));
//	thrust::device_ptr<float> thr_arp_ave = thrust::device_pointer_cast(d_arp_ave);
//
//	getARPMean << <(idx_1_len + block.x - 1) / block.x, block >> > (d_arp_ave, d_idx_1, d_arp, idx_1_len, WL, paras.range_num);
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	// idx_2 = find( APR_ave < low_threshold_gray )
//	thrust::device_vector<int> idx_2(idx_1_len);
//
//	auto end_idx_2 = thrust::copy_if(thrust::make_counting_iterator(0), thrust::make_counting_iterator(idx_1_len), \
//		thr_arp_ave, idx_2.begin(), thrust::placeholders::_1 < low_threshold_gray);
//	int idx_2_len = static_cast<int>(end_idx_2 - idx_2.begin());
//	idx_2.resize(idx_2_len);
//	int* d_idx_2 = thrust::raw_pointer_cast(idx_2.data());
//
//	if (idx_2_len != idx_1_len) {
//		// idx_1(idx_2) = 0
//		setNumInArray << <(idx_2_len + block.x - 1) / block.x, block >> > (d_idx_1, d_idx_2, 0, idx_2_len);
//		checkCudaErrors(cudaDeviceSynchronize());
//
//		int mean_idx = thrust::reduce(thrust::device, idx_1.begin(), idx_1.end(), 0, thrust::plus<int>()) / (idx_1_len - idx_2_len);
//		int shift_num = -(mean_idx - static_cast<int>(paras.range_num / 2));  // todo: +1???
//
//		// * circshift(d_data,[0, shiftnum])
//		circshiftFreq(d_data, paras.range_num, static_cast<float>(shift_num), paras.data_num, handles.handle, handles.plan_all_echo_c2c);
//	}
//
//	// * Free GPU Allocated Space
//	checkCudaErrors(cudaFree(d_hrrp));
//	checkCudaErrors(cudaFree(d_arp_ave));
//}


template <typename T>
void circshiftTime(T* d_data, int frag_len, int shift, int len)
{
	T* d_data_temp = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_data_temp, sizeof(T) * len));

	dim3 block(DEFAULT_THREAD_PER_BLOCK);  // block size

	circShiftKernel << <dim3((len + block.x - 1) / block.x), block >> > (d_data, d_data_temp, frag_len, shift, len);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(d_data, d_data_temp, sizeof(T) * len, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaFree(d_data_temp));
}


template <typename T>
__global__ void circShiftTimeKernel(T* d_in, T* d_out, int frag_len, int shift_num, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		//int base = static_cast<int>(tid / frag_len) * frag_len;
		//int offset = (tid % frag_len + shift_num) % frag_len;
		d_out[static_cast<int>(tid / frag_len) * frag_len + (tid % frag_len + shift_num) % frag_len] = d_in[tid];
	}
}


void circshiftFreq(cuComplex* d_data, int frag_len, float shift, int len, cublasHandle_t handle, cufftHandle plan_all_echo_c2c)
{
	dim3 block(DEFAULT_THREAD_PER_BLOCK);
	dim3 grid_one_frag((frag_len + block.x - 1) / block.x);  // grid size
	dim3 grid((len + block.x - 1) / block.x);  // grid size

	// fft
	checkCudaErrors(cufftExecC2C(plan_all_echo_c2c, d_data, d_data, CUFFT_FORWARD));

	// d_shift_vec = exp(-1j * 2 * pi * [0:N-1] * shift)
	cuComplex* d_shift_vec = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_shift_vec, sizeof(cuComplex) * frag_len));
	genFreqMovVec << <grid_one_frag, block >> > (d_shift_vec, shift, frag_len);
	checkCudaErrors(cudaDeviceSynchronize());

	// d_data = d_data * repmat(d_shift_vec)
	elementwiseMultiplyRep << <grid, block >> > (d_shift_vec, d_data, d_data, frag_len, len);
	checkCudaErrors(cudaDeviceSynchronize());

	// ifft
	checkCudaErrors(cufftExecC2C(plan_all_echo_c2c, d_data, d_data, CUFFT_INVERSE));
	float scale_ifft = 1 / static_cast<float>(frag_len);
	checkCudaErrors(cublasCsscal(handle, len, &scale_ifft, d_data, 1));

	checkCudaErrors(cudaFree(d_shift_vec));
}


__global__ void getARPMean(float* d_arp_ave, int* idx_1, float* arp, int idx_1_len, int WL, int range_num)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float temp_sum = 0;

	// [todo] possible optimization: assign a block to calculate mean value of a indice
	if (idx_1[tid] - WL / 2 >= 0 && idx_1[tid] + WL / 2 <= range_num - 1) {
		for (int index_ARP = idx_1[tid] - WL / 2; index_ARP <= idx_1[tid] + WL / 2; index_ARP++) {
			temp_sum += arp[index_ARP];
		}
		temp_sum /= WL + 1;
	}
	else if (idx_1[tid] - WL / 2 < 0) {
		for (int index_ARP = 0; index_ARP <= idx_1[tid] + WL / 2; index_ARP++) {
			temp_sum += arp[index_ARP];
		}
		temp_sum /= (static_cast<float>(WL) / 2 + idx_1[tid] + 1);
	}
	else if (idx_1[tid] + WL / 2 > range_num - 1) {
		for (int index_ARP = idx_1[tid] - WL / 2; index_ARP < range_num; index_ARP++) {
			temp_sum += arp[index_ARP];
		}
		temp_sum /= (static_cast<float>(range_num)-idx_1[tid] - (static_cast<float>(WL) / 2) - 1);
	}
	d_arp_ave[tid] = temp_sum;

}

