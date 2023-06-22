#include "common.cuh"


/* CUDAHandle Class */
void CUDAHandle::handleInit(const int& echo_num, const int& range_num)
{
	checkCudaErrors(cublasCreate(&handle));

	checkCudaErrors(cufftPlan1d(&plan_all_echo_c2c, range_num, CUFFT_C2C, echo_num));
	//checkCudaErrors(cufftPlan1d(&plan_one_echo_c2c, range_num, CUFFT_C2C, 1));
	//checkCudaErrors(cufftPlan1d(&plan_one_echo_r2c, range_num, CUFFT_R2C, 1));
	checkCudaErrors(cufftPlan1d(&plan_all_echo_r2c, range_num, CUFFT_R2C, echo_num));
	//checkCudaErrors(cufftPlan1d(&plan_one_echo_c2r, range_num, CUFFT_C2R, 1));
	checkCudaErrors(cufftPlan1d(&plan_all_echo_c2r, range_num, CUFFT_C2R, echo_num));

	// cuFFT data layout for applying fft to each column along first dimension
	int batch = RANGE_NUM_CUT;
	int rank = 1;
	int n[] = { echo_num };
	int inembed[] = { echo_num };
	int onembed[] = { echo_num };
	int istride = RANGE_NUM_CUT;
	int ostride = RANGE_NUM_CUT;
	int idist = 1;
	int odist = 1;
	checkCudaErrors(cufftPlanMany(&plan_all_range_c2c, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch));

	int fft_len = nextPow2(2 * echo_num - 1);
	n[0] = fft_len;
	inembed[0] = fft_len;
	onembed[0] = fft_len;
	checkCudaErrors(cufftPlanMany(&plan_all_range_c2c_czt, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch));

	checkCudaErrors(cufftPlan1d(&plan_all_echo_c2c_cut, RANGE_NUM_CUT, CUFFT_C2C, echo_num));
}


void CUDAHandle::handleDest()
{
	checkCudaErrors(cublasDestroy(handle));

	checkCudaErrors(cufftDestroy(plan_all_echo_c2c));
	//checkCudaErrors(cufftDestroy(plan_one_echo_c2c));
	//checkCudaErrors(cufftDestroy(plan_one_echo_r2c));
	checkCudaErrors(cufftDestroy(plan_all_echo_r2c));
	//checkCudaErrors(cufftDestroy(plan_one_echo_c2r));
	checkCudaErrors(cufftDestroy(plan_all_echo_c2r));

	checkCudaErrors(cufftDestroy(plan_all_range_c2c));
	checkCudaErrors(cufftDestroy(plan_all_range_c2c_czt));
	checkCudaErrors(cufftDestroy(plan_all_echo_c2c_cut));
}


void getMax(cublasHandle_t handle, float* d_vec, int len, int* h_max_idx, float* h_max_val)
{
	checkCudaErrors(cublasIsamax(handle, len, d_vec, 1, h_max_idx));
	--(*h_max_idx);  // cuBlas using 1-based indexing

	checkCudaErrors(cudaMemcpy(h_max_val, d_vec + *h_max_idx, sizeof(float), cudaMemcpyDeviceToHost));
}


void getMax(cublasHandle_t handle, cuComplex* d_vec, int len, int* h_max_idx, cuComplex* h_max_val)
{
	checkCudaErrors(cublasIcamax(handle, len, d_vec, 1, h_max_idx));
	--(*h_max_idx);  // cuBlas using 1-based indexing

	checkCudaErrors(cudaMemcpy(h_max_val, d_vec + *h_max_idx, sizeof(cuComplex), cudaMemcpyDeviceToHost));
}


void getMin(cublasHandle_t handle, float* d_vec, int len, int* min_idx, float* min_val)
{
	checkCudaErrors(cublasIsamin(handle, len, d_vec, 1, min_idx));
	--(*min_idx);  // cuBlas using 1-based indexing

	checkCudaErrors(cudaMemcpy(min_val, d_vec + *min_idx, sizeof(float) * 1, cudaMemcpyDeviceToHost));
}


__global__ void elementwiseAbs(cuComplex* a, float* abs, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		abs[tid] = cuCabsf(a[tid]);
	}
}


__global__ void elementwiseConj(cuComplex* d_data, cuComplex* d_data_conj, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		d_data_conj[tid] = cuConjf(d_data[tid]);
	}
}


__global__ void elementwiseMultiply(cuComplex* a, cuComplex* b, cuComplex* c, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		c[tid] = cuCmulf(a[tid], b[tid]);
	}
}


__global__ void elementwiseMultiply(cuDoubleComplex* a, cuDoubleComplex* b, cuDoubleComplex* c, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		c[tid] = cuCmul(a[tid], b[tid]);
	}
}


__global__ void elementwiseMultiply(float* a, float* b, float* c, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		c[tid] = a[tid] * b[tid];
	}
}


__global__ void elementwiseMultiply(float* a, cuComplex* b, cuComplex* c, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		c[tid] = make_cuComplex(a[tid] * b[tid].x, a[tid] * b[tid].y);
	}
}


__global__ void elementwiseMultiplyConjA(cuComplex* a, cuComplex* b, cuComplex* c, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		c[tid] = cuCmulf(cuConjf(a[tid]), b[tid]);
	}
}


__global__ void elementwiseMultiplyRep(cuComplex* a, cuComplex* b, cuComplex* c, int len_a, int len_b)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len_b) {
		c[tid] = cuCmulf(a[tid % len_a], b[tid]);
	}
}


__global__ void elementwiseMultiplyRep(cuDoubleComplex* a, cuDoubleComplex* b, cuDoubleComplex* c, int len_a, int len_b)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len_b) {
		c[tid] = cuCmul(a[tid % len_a], b[tid]);
	}
}


__global__ void elementwiseMultiplyRep(float* a, cuComplex* b, cuComplex* c, int len_a, int len_b)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len_b) {
		c[tid] = cuCmulf(make_cuComplex(a[tid % len_a], 0.0f), b[tid]);
	}
}


__global__ void elementwiseDiv(float* a, cuComplex* b, cuComplex* c, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		c[tid] = make_cuComplex(b[tid].x / a[tid], b[tid].y / a[tid]);
	}
}


__global__ void elementwiseDivRep(float* a, float* b, float* c, int len_a, int len_b)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len_b) {
		c[tid] = b[tid] / a[tid % len_a];
	}
}


__global__ void diagMulMat(cuComplex* d_diag, cuComplex* d_data, cuComplex* d_res, int cols, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < len) {
		d_res[tid] = cuCmulf(d_diag[static_cast<int>(tid / cols)], d_data[tid]);
	}
}


__global__ void diagMulMat(float* d_diag, cuComplex* d_data, cuComplex* d_res, int cols, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < len) {
		float tmp = d_diag[static_cast<int>(tid / cols)];
		d_res[tid] = make_cuComplex(tmp * d_data[tid].x, tmp * d_data[tid].y);
	}
}


__global__ void diagMulMat(double* d_diag, double* d_data, double* d_res, int cols, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < len) {
		d_res[tid] = d_diag[static_cast<int>(tid / cols)] * d_data[tid];
	}
}


__global__ void expJ(double* x, cuDoubleComplex* res, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		res[tid] = make_cuDoubleComplex(std::cos(x[tid]), std::sin(x[tid]));
	}
}


__global__ void genHammingVec(float* d_hamming, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		d_hamming[tid] = (0.54f - 0.46f * std::cos(2 * PI_FLT * static_cast<float>(tid) / (len - 1)));
	}
}


void genHammingVecInit(float* d_hamming, int range_num, float* d_hamming_echoes, int echo_num)
{
	dim3 block(DEFAULT_THREAD_PER_BLOCK);

	// * Adding hamming window in range dimension
	genHammingVec << <dim3((range_num + block.x - 1) / block.x), block >> > (d_hamming, range_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// * Adding hamming window in range dimension
	genHammingVec << <dim3((echo_num + block.x - 1) / block.x), block >> > (d_hamming_echoes, echo_num);
	checkCudaErrors(cudaDeviceSynchronize());
}


//template <typename T>
//__global__ void getMaxIdx(const T* data, const int dsize, int* result)
//{
//
//	__shared__ volatile T   vals[nTPB];
//	__shared__ volatile int idxs[nTPB];
//	__shared__ volatile int last_block;
//	int idx = threadIdx.x + blockDim.x * blockIdx.x;
//	last_block = 0;
//	T   my_val = FLOAT_MIN;
//	int my_idx = -1;
//	// sweep from global memory
//	while (idx < dsize) {
//		if (data[idx] > my_val) { my_val = data[idx]; my_idx = idx; }
//		idx += blockDim.x * gridDim.x;
//	}
//	// populate shared memory
//	vals[threadIdx.x] = my_val;
//	idxs[threadIdx.x] = my_idx;
//	__syncthreads();
//	// sweep in shared memory
//	for (int i = (nTPB >> 1); i > 0; i >>= 1) {
//		if (threadIdx.x < i)
//			if (vals[threadIdx.x] < vals[threadIdx.x + i]) { vals[threadIdx.x] = vals[threadIdx.x + i]; idxs[threadIdx.x] = idxs[threadIdx.x + i]; }
//		__syncthreads();
//	}
//	// perform block-level reduction
//	if (!threadIdx.x) {
//		blk_vals[blockIdx.x] = vals[0];
//		blk_idxs[blockIdx.x] = idxs[0];
//		if (atomicAdd(&blk_num, 1) == gridDim.x - 1) // then I am the last block
//			last_block = 1;
//	}
//	__syncthreads();
//	if (last_block) {
//		idx = threadIdx.x;
//		my_val = FLOAT_MIN;
//		my_idx = -1;
//		while (idx < gridDim.x) {
//			if (blk_vals[idx] > my_val) { my_val = blk_vals[idx]; my_idx = blk_idxs[idx]; }
//			idx += blockDim.x;
//		}
//		// populate shared memory
//		vals[threadIdx.x] = my_val;
//		idxs[threadIdx.x] = my_idx;
//		__syncthreads();
//		// sweep in shared memory
//		for (int i = (nTPB >> 1); i > 0; i >>= 1) {
//			if (threadIdx.x < i)
//				if (vals[threadIdx.x] < vals[threadIdx.x + i]) { vals[threadIdx.x] = vals[threadIdx.x + i]; idxs[threadIdx.x] = idxs[threadIdx.x + i]; }
//			__syncthreads();
//		}
//		if (!threadIdx.x)
//			*result = idxs[0];
//	}
//}


__global__ void maxCols(float* d_data, float* d_max_clos, int rows, int cols)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int nTPB = blockDim.x;

	// [todo] Possible optimization:  halve the number of threads and size of shared memory assigned for each block. (only if echo number lesser than 1024)
	// Perform a reduction within the block to compute the final maximum value.
	// sdata_max_cols_int store the index of the maximum value in each block.
	extern __shared__ int sdata_max_cols_int[];
	sdata_max_cols_int[tid] = tid * cols + bid;
	__syncthreads();

	for (int s = (nTPB >> 1); s > 0; s >>= 1) {
		if (tid < s) {
			if (d_data[sdata_max_cols_int[tid]] < d_data[sdata_max_cols_int[tid + s]]) {
				sdata_max_cols_int[tid] = sdata_max_cols_int[tid + s];
			}
		}
		__syncthreads();
	}

	if (tid == 0) {
		d_max_clos[bid] = d_data[sdata_max_cols_int[0]];
	}
}


__global__ void sumCols(float* d_data, float* d_sum_clos, int rows, int cols)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int nTPB = blockDim.x;

	// [todo] Possible optimization:  halve the number of threads and size of shared memory assigned for each block. (only if echo number lesser than 1024)
	// Perform a reduction within the block to compute the final sum
	extern __shared__ float sdata_sum_cols_flt[];
	sdata_sum_cols_flt[tid] = d_data[tid * cols + bid];
	__syncthreads();

	for (int s = (nTPB >> 1); s > 0; s >>= 1) {
		if (tid < s) {
			sdata_sum_cols_flt[tid] += sdata_sum_cols_flt[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		d_sum_clos[bid] = sdata_sum_cols_flt[0];
	}
}


/// <summary>
/// Getting the max element index of every single row in matrix d_data.
/// Each block is responsible for the calculation of a single row.
/// Kernel configuration requirements:
/// (1) block_number == rows
/// (2) shared_memory_number == thread_per_block == cols
/// </summary>
/// <param name="d_data"></param>
/// <param name="d_max_rows"></param>
/// <param name="rows"></param>
/// <param name="cols"></param>
/// <param name="extra"></param>
/// <returns></returns>
__global__ void maxRowsIdx(float* d_data, int* d_max_rows_idx, int rows, int cols, float extra)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int nTPB = blockDim.x;

	int t_max_idx = tid;
	for (int i = tid + nTPB; i < cols; i += nTPB) {
		t_max_idx = (d_data[bid * cols + i] > d_data[bid * cols + t_max_idx]) ? i : t_max_idx;
	}

	// Perform a reduction within the block to compute the final sum
	extern __shared__ int sdata_max_rows_int[];
	sdata_max_rows_int[tid] = t_max_idx;
	__syncthreads();

	for (int s = (nTPB >> 1); s > 0; s >>= 1) {
		if (tid < s) {
			sdata_max_rows_int[tid] = (d_data[bid * cols + sdata_max_rows_int[tid]] > d_data[bid * cols + sdata_max_rows_int[tid + s]]) ? sdata_max_rows_int[tid] : sdata_max_rows_int[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		d_max_rows_idx[bid] = sdata_max_rows_int[0];
	}
}


__global__ void sumRows(cuComplex* d_data, cuComplex* d_sum_rows, int rows, int cols)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int nTPB = blockDim.x;

	cuComplex t_sum = make_cuComplex(0.0f, 0.0f);
	for (int i = tid; i < cols; i += nTPB) {
		t_sum = cuCaddf(t_sum, d_data[bid * cols + i]);
	}

	// Perform a reduction within the block to compute the final sum
	extern __shared__ cuComplex sdata_sum_rows_com_flt[];
	sdata_sum_rows_com_flt[tid] = t_sum;
	__syncthreads();

	for (int s = (nTPB >> 1); s > 0; s >>= 1) {
		if (tid < s) {
			sdata_sum_rows_com_flt[tid] = cuCaddf(sdata_sum_rows_com_flt[tid], sdata_sum_rows_com_flt[tid + s]);
		}
		__syncthreads();
	}

	if (tid == 0) {
		d_sum_rows[bid] = sdata_sum_rows_com_flt[0];
	}
}


__global__ void sumRows(float* d_data, float* d_sum_rows, int rows, int cols)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int nTPB = blockDim.x;

	float t_sum = 0.0f;
	for (int i = tid; i < cols; i += nTPB) {
		t_sum = t_sum + d_data[bid * cols + i];
	}

	// Perform a reduction within the block to compute the final sum
	extern __shared__ float sdata_sum_rows_flt[];
	sdata_sum_rows_flt[tid] = t_sum;
	__syncthreads();

	for (int s = (nTPB >> 1); s > 0; s >>= 1) {
		if (tid < s) {
			sdata_sum_rows_flt[tid] = sdata_sum_rows_flt[tid] + sdata_sum_rows_flt[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		d_sum_rows[bid] = sdata_sum_rows_flt[0];
	}
}


void cutRangeProfile(cuComplex* d_data, cuComplex* d_data_cut, \
	const int& cols, const int& cols_cut, const int& data_num_cut, const cublasHandle_t& handle)
{
	dim3 block(DEFAULT_THREAD_PER_BLOCK);  // block size

	// find the index of the maximum value in the first echo
	int range_abs_max_idx = 0;
	cuComplex range_abs_max_val = make_cuComplex(0.0f, 0.0f);
	getMax(handle, d_data, cols, &range_abs_max_idx, &range_abs_max_val);

	int offset = MAX(range_abs_max_idx - cols_cut / 2, 0);
	offset = MIN(offset, cols - cols_cut);

	cutRangeProfileHelper << <(data_num_cut + block.x - 1) / block.x, block >> > (d_data, d_data_cut, cols, cols_cut, offset, data_num_cut);
	checkCudaErrors(cudaDeviceSynchronize());
}


__global__ void cutRangeProfileHelper(cuComplex* d_in, cuComplex* d_out, int cols, int cols_cut, int offset, int data_num_cut)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < data_num_cut) {
		d_out[tid] = d_in[(tid / cols_cut) * cols + offset + (tid % cols_cut)];
	}
}


int nextPow2(int N) {
	int n = 1;
	while (N >> 1) {
		n = n << 1;
		N = N >> 1;
	}
	n = n << 1;
	return n;
}


__global__ void setNumInArray(int* d_data, int* d_index, int val, int d_index_len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < d_index_len) {
		d_data[d_index[tid]] = val;
	}
}


void getHRRP(cuComplex* d_hrrp, cuComplex* d_data, float* d_hamming, const RadarParameters& paras, const DATA_TYPE& data_type, const CUDAHandle& handles)
{
	dim3 block(DEFAULT_THREAD_PER_BLOCK);  // block size

	switch (data_type) {
	case DATA_TYPE::IFDS: {
		// copy data to d_hrrp
		checkCudaErrors(cudaMemcpy(d_hrrp, d_data, sizeof(cuComplex) * paras.data_num, cudaMemcpyDeviceToDevice));

		// fftshift
		ifftshiftRows << <dim3(((paras.range_num / 2) + block.x - 1) / block.x, paras.echo_num), block >> > (d_hrrp, paras.range_num);
		checkCudaErrors(cudaDeviceSynchronize());

		// fft
		checkCudaErrors(cufftExecC2C(handles.plan_all_echo_c2c, d_hrrp, d_hrrp, CUFFT_FORWARD));

		// fftshift
		ifftshiftRows << <dim3(((paras.range_num / 2) + block.x - 1) / block.x, paras.echo_num), block >> > (d_hrrp, paras.range_num);
		checkCudaErrors(cudaDeviceSynchronize());

		break;
	}
	case DATA_TYPE::STRETCH: {
		// adding hamming window to each echo
		elementwiseMultiplyRep << <dim3((paras.data_num + block.x - 1) / block.x), block >> > (d_hamming, d_data, d_data, paras.range_num, paras.data_num);
		checkCudaErrors(cudaDeviceSynchronize());

		// fft
		checkCudaErrors(cufftExecC2C(handles.plan_all_echo_c2c, d_data, d_hrrp, CUFFT_FORWARD));

		// fftshift
		ifftshiftRows << <dim3(((paras.range_num / 2) + block.x - 1) / block.x, paras.echo_num), block >> > (d_hrrp, paras.range_num);
		checkCudaErrors(cudaDeviceSynchronize());

		break;
	}
	default:
		break;
	}
}


void multiThreadIFDS(cuComplex* d_data, int16_t* dataAD, std::complex<float>* dataIQ, std::vector<std::ifstream>& ifs_vec, \
	const int startIdx, const int endIdx, const int dataIQ_size, const int dataAD_size, const int frame_num, const int frame_len, \
	const RadarParameters& paras, const vec1D_INT& dataWFileSn, const vec1D_DBL& dataNOut)
{
	// Initializing object of pulse compression for a single thread.
	pulseCompression pc(nextPow2(dataIQ_size), dataIQ_size, RANGE_NUM_IFDS_PC, paras);

	for (int i = startIdx; i < endIdx; ++i) {
		int file_idx = dataWFileSn[i] / frame_num;

		// Read data
		ifs_vec[file_idx].seekg((dataWFileSn[i] - file_idx * frame_num) * frame_len + 256, std::ifstream::beg);
		ifs_vec[file_idx].read((char*)dataAD, dataAD_size * sizeof(int16_t));

		for (int j = 0; (j + 1) < dataAD_size; j += 2) {
			dataIQ[j / 2] = std::complex<float>(static_cast<float>(dataAD[j]), static_cast<float>(dataAD[j + 1]));
		}

		// Pulse compression
		pc.pulseCompressionbyFFT(d_data + i * RANGE_NUM_IFDS_PC, dataIQ, dataNOut[paras.echo_num + dataWFileSn[i]]);
	}
}


void multiThreadSTRETCH(std::complex<float>* h_data, cuComplex* d_data, int16_t* dataAD, std::vector<std::ifstream>& ifs_vec, \
	const int startIdx, const int endIdx, const int dataAD_size, const int frame_num, const int frame_len, \
	const RadarParameters& paras, const vec1D_INT& dataWFileSn, const vec1D_DBL& dataNOut)
{
	for (int i = startIdx; i < endIdx; ++i) {
		int file_idx = dataWFileSn[i] / frame_num;

		ifs_vec[file_idx].seekg((dataWFileSn[i] - file_idx * frame_num) * frame_len + 256, std::ifstream::beg);
		ifs_vec[file_idx].read((char*)dataAD, dataAD_size * sizeof(int16_t));

		for (int j = 0; (j + 1) < dataAD_size; j += 2) {
			h_data[i * paras.range_num + (j / 2)] = std::complex<float>(static_cast<float>(dataAD[j]), static_cast<float>(dataAD[j + 1]));
		}

		// transfer data(one echo) to device asynchronously
		checkCudaErrors(cudaMemcpyAsync(d_data + i * paras.range_num, h_data + i * paras.range_num, sizeof(cuComplex) * paras.range_num, cudaMemcpyHostToDevice));
	}
}


/// <summary>
/// 
/// </summary>
/// <param name="d_tk"></param>
/// <param name="Tp"></param>
/// <param name="constant"> 1 - 2 * velocity / c </param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void genTkPulseCompression(float* d_tk, float Tp, float constant, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < len) {
		d_tk[tid] = constant * (Tp / -2.0f + static_cast<float>(tid) * Tp / (static_cast<float>(len) - 1.0f));
	}
}


/// <summary>
/// 
/// </summary>
/// <param name="d_ref"></param>
/// <param name="d_tk"></param>
/// <param name="d_hamming"></param>
/// <param name="constant_1"> -2 * PI_FLT * F0 * v2 / _v2 </param>
/// <param name="constant_2"> PI_FLT * Band / Taup </param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void genRefPulseCompression(cuComplex* d_ref, float* d_tk, float* d_hamming, float constant_1, float constant_2, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < len) {
		//temp2 = -2 * 3.14159265 * F0 * v2 * t[i] / temp + 3.14159265 * Band / Taup * pow(t[i], 2);  //3.14159265
		float tmp = (constant_1 + constant_2 * d_tk[tid]) * d_tk[tid];
		d_ref[tid] = make_cuComplex(d_hamming[tid] * std::cos(tmp), d_hamming[tid] * std::sin(tmp));
		
		//if (tid == 0) {
		//	printf("[genRefPulseCompression] %f %f %f %f %f %f\n", d_hamming[tid], tmp, std::cos(tmp), std::sin(tmp), d_ref[tid].x, d_ref[tid].y);
		//}
		//if (tid == 100) {
		//	printf("[genRefPulseCompression] %f %f %f %f %f %f\n", d_hamming[tid], tmp, std::cos(tmp), std::sin(tmp), d_ref[tid].x, d_ref[tid].y);
		//}
		//if (tid == 2399999) {
		//	printf("[genRefPulseCompression] %f %f %f %f %f %f\n", d_hamming[tid], tmp, std::cos(tmp), std::sin(tmp), d_ref[tid].x, d_ref[tid].y);
		//}
		//if (tid == 2400000) {
		//	printf("[genRefPulseCompression] %f %f %f %f %f %f\n", d_hamming[tid], tmp, std::cos(tmp), std::sin(tmp), d_ref[tid].x, d_ref[tid].y);
		//}
	}
}


/* pulseCompression Class */
pulseCompression::pulseCompression(const int& NFFT, const int& dataIQ_len, const int& range_num_ifds_pc, const RadarParameters& paras)
{
	m_NFFT = NFFT;
	m_dataIQ_len = dataIQ_len;
	m_range_num_ifds_pc = range_num_ifds_pc;
	m_paras = paras;
	m_sampling_num = static_cast<int>(m_paras.Fs * m_paras.Tp);

	// Initializing cuda handle
	checkCudaErrors(cublasCreate(&handle));
	checkCudaErrors(cufftPlan1d(&plan_pc_echo_c2c, m_NFFT, CUFFT_C2C, 1));

	// Allocate memory in device
	checkCudaErrors(cudaMalloc((void**)&d_dataIQ, sizeof(cuComplex) * m_NFFT));
	checkCudaErrors(cudaMalloc((void**)&d_hamming, sizeof(float) * m_sampling_num));
	checkCudaErrors(cudaMalloc((void**)&d_tk, sizeof(float) * m_sampling_num));
	checkCudaErrors(cudaMalloc((void**)&d_ref, sizeof(cuComplex) * m_NFFT));

	// Generate hamming window vector
	dim3 block(DEFAULT_THREAD_PER_BLOCK);
	genHammingVec << <dim3((m_sampling_num + block.x - 1) / block.x), block >> > (d_hamming, m_sampling_num);
	checkCudaErrors(cudaDeviceSynchronize());
}


pulseCompression::~pulseCompression()
{
	// Free cuda handle
	checkCudaErrors(cublasDestroy(handle));
	checkCudaErrors(cufftDestroy(plan_pc_echo_c2c));

	// Free allocated memory in device
	checkCudaErrors(cudaFree(d_dataIQ));
	checkCudaErrors(cudaFree(d_hamming));
	checkCudaErrors(cudaFree(d_tk));
	checkCudaErrors(cudaFree(d_ref));
}


void pulseCompression::pulseCompressionbyFFT(cuComplex* d_dataW_echo, \
	const std::complex<float>* h_dataIQ_echo, const double velocity_echo)
{
	dim3 block(DEFAULT_THREAD_PER_BLOCK);

	// h_dataIQ_echo(host to device)
	checkCudaErrors(cudaMemcpy(d_dataIQ, h_dataIQ_echo, sizeof(cuComplex) * m_dataIQ_len, cudaMemcpyHostToDevice));

	// set extra memory out of range to 0
	cudaMemset(d_dataIQ + m_dataIQ_len, 0, (m_NFFT - m_dataIQ_len) * sizeof(cuComplex));
	cudaMemset(d_ref + m_sampling_num, 0, (m_NFFT - m_sampling_num) * sizeof(cuComplex));

	// display
	//std::cout << velocity_echo << std::endl;
	//std::cout << "d_dataIQ" << std::endl;
	//dDataDisp(d_dataIQ + 0, 1, 10);
	//dDataDisp(d_dataIQ + 100, 1, 10);
	//dDataDisp(d_dataIQ + 1000, 1, 10);

	// Generating reference signal
	float v2 = 2 * static_cast<float>(velocity_echo) / LIGHT_SPEED;
	float _v2 = 1 - v2;
	float constant_1 = static_cast<float>(-2 * PI_FLT * m_paras.fc * v2 / _v2);
	float constant_2 = static_cast<float>(PI_FLT * m_paras.band_width / m_paras.Tp);
	float scale_ifft = 1.0f / m_NFFT;

	// display
	//std::cout << "d_hamming: " << std::endl;
	//dDataDisp(d_hamming + 0, 1, 10);
	//dDataDisp(d_hamming + 100, 1, 10);
	//dDataDisp(d_hamming + 2300000, 1, 10);

	genTkPulseCompression << <dim3((m_sampling_num + block.x - 1) / block.x), block >> > (d_tk, static_cast<float>(m_paras.Tp), _v2, m_sampling_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// display
	//std::cout << "d_tk: " << std::endl;
	//dDataDisp(d_tk + 0, 1, 10);
	//dDataDisp(d_tk + 100, 1, 10);
	//dDataDisp(d_tk + 2300000, 1, 10);

	genRefPulseCompression << <dim3((m_sampling_num + block.x - 1) / block.x), block >> > (d_ref, d_tk, d_hamming, constant_1, constant_2, m_sampling_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// display
	//std::cout << "d_ref: " << std::endl;
	//dDataDisp(d_ref + 0, 1, 10);
	//dDataDisp(d_ref + 100, 1, 10);
	//dDataDisp(d_ref + 2400000, 1, 10);

	// FFT(signal and reference)
	checkCudaErrors(cufftExecC2C(plan_pc_echo_c2c, d_dataIQ, d_dataIQ, CUFFT_FORWARD));
	checkCudaErrors(cufftExecC2C(plan_pc_echo_c2c, d_ref, d_ref, CUFFT_FORWARD));

	// display
	//std::cout << "d_dataIQ after fft" << std::endl;
	//dDataDisp(d_dataIQ + 0, 1, 10);
	//dDataDisp(d_dataIQ + 100, 1, 10);
	//dDataDisp(d_dataIQ + 2400000, 1, 10);
	//std::cout << "d_ref after fft" << std::endl;
	//dDataDisp(d_ref + 0, 1, 10);
	//dDataDisp(d_ref + 100, 1, 10);
	//dDataDisp(d_ref + 2400000, 1, 10);

	// Conjunction multiply
	elementwiseMultiplyConjA << <dim3((m_NFFT + block.x - 1) / block.x), block >> > (d_ref, d_dataIQ, d_dataIQ, m_NFFT);
	checkCudaErrors(cudaDeviceSynchronize());

	// display
	//std::cout << "d_dataIQ after conjMul" << std::endl;
	//dDataDisp(d_dataIQ + 0, 1, 10);
	//dDataDisp(d_dataIQ + 100, 1, 10);
	//dDataDisp(d_dataIQ + 2400000, 1, 10);

	// IFFT(signal)
	checkCudaErrors(cufftExecC2C(plan_pc_echo_c2c, d_dataIQ, d_dataIQ, CUFFT_INVERSE));
	checkCudaErrors(cublasCsscal(handle, m_NFFT, &scale_ifft, d_dataIQ, 1));

	// display
	//std::cout << "d_dataIQ after ifft" << std::endl;
	//dDataDisp(d_dataIQ + 0, 1, 10);
	//dDataDisp(d_dataIQ + 100, 1, 10);
	//dDataDisp(d_dataIQ + 2400000, 1, 10);

	// Cut range profile
	cutRangeProfile(d_dataIQ, d_dataW_echo, m_NFFT, m_range_num_ifds_pc, m_range_num_ifds_pc, handle);
}


/* ioOperation Class */
void ioOperation::ioInit(std::string* INTERMEDIATE_DIR, const std::string& dir_path, const POLAR_TYPE& polar_type, const DATA_TYPE& data_type)
{
	m_dir_path = fs::absolute(dir_path).string();
	m_polar_type = polar_type;
	m_data_type = data_type;
	m_file_vec.clear();

	// Check if 'dir_path' is a valid directory
	if (!fs::is_directory(m_dir_path)) {
		std::cout << "[ioInit/ERROR] Invalid directory path!\n";
		return;
	}

	// Construct the file name pattern based on 'polar_type'
	std::string file_pattern;
	switch (m_data_type) {
	case DATA_TYPE::IFDS:
		file_pattern = (m_polar_type == POLAR_TYPE::LHP) ? R"(.*01_1100.*\.wbd)" : R"(.*01_1101.*\.wbd)";  // regex matches "*01_1100*.wbd" and "*01_1101*.wbd"
		break;
	case DATA_TYPE::STRETCH:
		file_pattern = (m_polar_type == POLAR_TYPE::LHP) ? R"(.*00_1100\.wbd)" : R"(.*00_1101\.wbd)";  // regex matches "*00_1100.wbd" and "*00_1101.wbd"
		break;
	default:
		std::cout << "[ioInit/ERROR] Invalid data type!\n";
		break;
	}
	std::regex regex_pattern(file_pattern);

	// Iterate over the directory and find files matching the pattern
	for (const auto& entry : fs::directory_iterator(m_dir_path)) {
		std::string file_name = fs::path(entry).string();
		if (entry.is_regular_file() && std::regex_match(file_name, regex_pattern)) {
			m_file_vec.push_back(file_name);
		}
	}

	if (m_file_vec.size() == 0) {
		std::cout << "[ioInit/ERROR] No valid file in the given directory!\n";
		return;
	}

	// Assign INTERMEDIATE_DIR
	*INTERMEDIATE_DIR = m_dir_path + std::string("\\intermediate\\");
}


int ioOperation::getSystemParas(RadarParameters* paras, int* frame_len, int* frame_num)
{
	std::ifstream ifs;
	ifs.open(m_file_vec[0], std::ios_base::in | std::ios_base::binary);
	if (!ifs) {
		std::cout << "[getSystemParas/WARN] Cannot open file " << m_file_vec[0] << " !\n";
		return EXIT_FAILURE;
	}

	ifs.seekg(0, std::ifstream::beg);
	uint32_t temp[36]{};
	ifs.read((char*)&temp, sizeof(uint32_t) * 36);  // 144 bytes in total

	*frame_len = static_cast<int>(temp[4]) * 4;  // length of frame, including frame head and orthogonal demodulation data.(unit: Byte)
	paras->fc = static_cast<long long>(temp[12]) * 1000000LL;  // signal carrier frequency
	paras->band_width = static_cast<long long>(temp[13]) * 1000000LL;  // signal band width
	paras->Tp = static_cast<double>(temp[15]) / 1000000.0;  // pulse width

	// sampling frequency
	switch (m_data_type)
	{
	case DATA_TYPE::IFDS:
		paras->Fs = static_cast<long long>(temp[17] % static_cast<uint32_t>(1 << 16)) * 1000000LL / 2LL;
		break;
	case DATA_TYPE::STRETCH:
		paras->Fs = static_cast<long long>(temp[17] % static_cast<uint32_t>(1 << 16)) * 1000000LL;
		break;
	default:
		std::cout << "[getSystemParas/ERROR] Invalid data type!\n";
		break;
	}

	*frame_num = static_cast<int>(fs::file_size(fs::path(m_file_vec[0]))) / *frame_len;  // total frame number in file

	ifs.close();

	return EXIT_SUCCESS;
}


int ioOperation::readKuIFDSAllNB(vec1D_DBL* dataN, vec1D_FLT* turnAngle, \
	const RadarParameters& paras, const int& frame_len, const int& frame_num)
{
	std::vector<std::ifstream> ifs_vec(m_file_vec.size());
	for (int i = 0; i < m_file_vec.size(); ++i) {
		ifs_vec[i].open(m_file_vec[i], std::ios_base::in | std::ios_base::binary);
		if (!ifs_vec[i]) {
			std::cout << "[readKuIFDSALLNBStretch/WARN] Cannot open file " << m_file_vec[i] << " !\n";
			return EXIT_FAILURE;
		}
	}

	int frame_num_total = frame_num * static_cast<int>(m_file_vec.size());

	dataN->resize(frame_num_total * 4);

	vec1D_FLT azimuthVec(frame_num_total);
	vec1D_FLT pitchingVec(frame_num_total);

	uint32_t headerData[11]{};

	double range = 0;  // unit: m
	double velocity = 0;  // unit: m/s
	double azimuth = 0;
	double pitching = 0;

	int file_idx = 0;  // file need to read
	for (int i = 0; i < frame_num_total; i++) {

		file_idx = i / frame_num;

		ifs_vec[file_idx].seekg((i - file_idx * frame_num) * frame_len + 48, std::ifstream::beg);
		ifs_vec[file_idx].read((char*)&headerData, sizeof(uint32_t) * 11);

		range = static_cast<double>(headerData[7]) * 0.1;
		velocity = static_cast<double>(headerData[8]);
		azimuth = static_cast<double>(headerData[9]);
		pitching = static_cast<double>(headerData[10]);

		velocity = (velocity - (velocity > std::pow(2, 31) ? std::pow(2, 32) : 0)) * 0.1;

		azimuth = (azimuth - (azimuth > std::pow(2, 31) ? std::pow(2, 32) : 0)) * (360.0 / std::pow(2, 24));
		azimuth += (azimuth < 0 ? 360.0 : 0);

		pitching = (pitching - (pitching > std::pow(2, 31) ? std::pow(2, 32) : 0)) * (360.0 / std::pow(2, 24));
		pitching += (pitching < 0 ? 360.0 : 0);

		dataN->at(i + 0 * frame_num_total) = range;
		dataN->at(i + 1 * frame_num_total) = velocity;
		dataN->at(i + 2 * frame_num_total) = azimuth;
		dataN->at(i + 3 * frame_num_total) = pitching;

		azimuthVec[i] = static_cast<float>(azimuth);
		pitchingVec[i] = static_cast<float>(pitching);
	}

	turnAngleLine(turnAngle, azimuthVec, pitchingVec);

	for (int i = 0; i < m_file_vec.size(); ++i) {
		ifs_vec[i].close();
	}

	return EXIT_SUCCESS;
}


int ioOperation::turnAngleLine(vec1D_FLT* turnAngle, const vec1D_FLT& azimuth, const vec1D_FLT& pitching) {

	vec1D_INT idx;
	int pitchingSize = static_cast<int>(pitching.size());
	for (int i = 0; i < pitchingSize - 1; ++i) {
		if (std::abs(pitching[i + 1] - pitching[i]) > 0.2) {
			idx.push_back(i);
		}
	}

	vec1D_INT blkBeginNum;
	vec1D_INT blkEndNum;
	vec1D_INT blkLen;
	int idxSize = idx.empty() ? 1 : static_cast<int>(idx.size());

	blkBeginNum.insert(blkBeginNum.cend(), -1);
	blkBeginNum.insert(blkBeginNum.cend(), idx.begin(), idx.end());
	int blkSize = static_cast<int>(blkBeginNum.size());
	std::for_each(blkBeginNum.begin(), blkBeginNum.end(), [](int& x) {x++; });

	blkEndNum.insert(blkEndNum.cend(), idx.begin(), idx.end());
	blkEndNum.insert(blkEndNum.cend(), pitchingSize - 1);

	blkLen.assign(blkSize, 0);
	std::transform(blkEndNum.cbegin(), blkEndNum.cend(), blkBeginNum.cbegin(), blkLen.begin(), [](const int& end, const int& begin) {return end - begin + 1; });

	turnAngle->assign(pitchingSize, 0);
	for (int blkIdx = 0; blkIdx < blkSize; ++blkIdx) {
		int N = blkLen[blkIdx];
		int stride = (N < 21) ? 1 : 20;
		for (int i = stride; i < N; i += stride) {
			int currentPulseNum = blkBeginNum[blkIdx] + i;
			float azimuth1 = azimuth[currentPulseNum - stride];
			float azimuth2 = azimuth[currentPulseNum];
			float pitching1 = pitching[currentPulseNum - stride];
			float pitching2 = pitching[currentPulseNum];
			float turnAngleSingle = getTurnAngle(azimuth1, pitching1, azimuth2, pitching2);
			turnAngle->at(currentPulseNum) = turnAngle->at(currentPulseNum - stride) + turnAngleSingle;  // angle superposition
		}
		int turnAngleSize = static_cast<int>(turnAngle->size());
		for (int i = 0; i < turnAngleSize; ++i) {
			turnAngle->at(i) = std::abs(turnAngle->at(i));
		}
		if (N >= 21) {
			vec1D_INT x = [=]() {
				vec1D_INT v;
				for (int i = 0; (i + stride) <= N; i += stride) {
					v.push_back(i);
				}
				return v;
			}();  // todo: range generate
			vec1D_FLT Y = [=]() {
				vec1D_FLT v;
				int xSize = static_cast<int>(x.size());
				for (int i = 0; i < xSize; ++i) {  // interpolation movement
					v.push_back(turnAngle->at(x[i]));
				}
				return v;
			}();
			vec1D_FLT turnAngleInterp = [=]() {
				vec1D_FLT v;
				for (int i = 0; i < N; ++i) {
					v.push_back(interpolate(x, Y, i, false));
				}
				return v;
			}();
			turnAngle->erase(turnAngle->cbegin() + blkBeginNum[blkIdx], turnAngle->cbegin() + blkEndNum[blkIdx] + 1);
			turnAngle->insert(turnAngle->cbegin() + blkBeginNum[blkIdx], turnAngleInterp.cbegin(), turnAngleInterp.cend());
		}

		if (blkIdx > 0) {
			for (int i = blkBeginNum[blkIdx]; i <= blkEndNum[blkIdx]; ++i) {
				turnAngle->at(i) += turnAngle->at(blkEndNum[blkIdx - 1]);
			}
		}
	}

	return EXIT_SUCCESS;
}


float ioOperation::getTurnAngle(const float& azimuth1, const float& pitching1, const float& azimuth2, const float& pitching2) {
	vec1D_FLT vec_1({ std::sin(pitching1 / 180 * PI_FLT), \
		std::cos(pitching1 / 180 * PI_FLT) * std::cos(azimuth1 / 180 * PI_FLT), \
		std::cos(pitching1 / 180 * PI_FLT) * std::sin(azimuth1 / 180 * PI_FLT) });

	vec1D_FLT vec_2({ std::sin(pitching2 / 180 * PI_FLT), \
		std::cos(pitching2 / 180 * PI_FLT) * std::cos(azimuth2 / 180 * PI_FLT), \
		std::cos(pitching2 / 180 * PI_FLT) * std::sin(azimuth2 / 180 * PI_FLT) });

	float ret = std::acos(vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1] + vec_1[2] * vec_2[2]) / PI_FLT * 180;

	return ret;
}


float ioOperation::interpolate(const vec1D_INT& xData, const vec1D_FLT& yData, const int& x, const bool& extrapolate) {
	int size = static_cast<int>(xData.size());

	int i = 0;  // find left end of interval for interpolation
	if (x >= xData[size - 2]) {  // special case: beyond right end
		i = size - 2;
	}
	else {
		while (x > xData[i + 1]) i++;
	}
	float xL = static_cast<float>(xData[i]);
	float yL = yData[i];
	float xR = static_cast<float>(xData[i + 1]);
	float yR = yData[i + 1];  // points on either side (unless beyond ends)
	if (!extrapolate) {  // if beyond ends of array and not extrapolating
		if (x < xL) yR = yL;
		if (x > xR) yL = yR;
	}

	float dydx = (yR - yL) / (xR - xL);  // gradient

	return yL + dydx * (x - xL);  // linear interpolation
}


int ioOperation::getSignalData(std::complex<float>* h_data, cuComplex* d_data, double* d_velocity, \
	const RadarParameters& paras, const vec1D_DBL& dataNOut, const int& frame_len, const int& frame_num, const vec1D_INT& dataWFileSn)
{
	std::vector<std::ifstream> ifs_vec(m_file_vec.size());
	for (int i = 0; i < m_file_vec.size(); ++i) {
		ifs_vec[i].open(m_file_vec[i], std::ios_base::in | std::ios_base::binary);
		if (!ifs_vec[i]) {
			std::cout << "[readKuIFDSALLNBStretch/WARN] Cannot open file " << m_file_vec[i] << " !\n";
			return EXIT_FAILURE;
		}
	}

	switch (m_data_type) {
	case DATA_TYPE::IFDS: {
		// Split the workload among CPU threads
		int numThreads = 8;
		int subsetSize = paras.echo_num / numThreads;

		// Create a vector to hold the CPU threads
		std::vector<std::thread> threads(numThreads);

		// Create dataAD and dataIQ arrays for each thread to avoid data races
		int dataAD_size = (frame_len - 256) / 2;
		int dataIQ_size = dataAD_size / 2;

		std::vector<int16_t*> dataADThreads(numThreads);
		std::vector<std::complex<float>*> dataIQThreads(numThreads);
		for (int t = 0; t < numThreads; ++t) {
			dataADThreads[t] = new int16_t[dataAD_size];
			dataIQThreads[t] = new std::complex<float>[dataIQ_size];
		}

		// Launch the CPU threads
		for (int t = 0; t < numThreads; ++t) {
			int startIdx = t * subsetSize;
			int endIdx = (t == numThreads - 1) ? paras.echo_num : (startIdx + subsetSize);

			threads[t] = std::thread(multiThreadIFDS, d_data, dataADThreads[t], dataIQThreads[t], std::ref(ifs_vec), \
				startIdx, endIdx, dataIQ_size, dataAD_size, frame_num, frame_len, \
				std::ref(paras), std::ref(dataWFileSn), std::ref(dataNOut));
		}

		// Wait for all the threads to finish
		for (int t = 0; t < numThreads; ++t) {
			threads[t].join();
		}

		// Clean up the dynamically allocated arrays
		for (int t = 0; t < numThreads; ++t) {
			delete[] dataADThreads[t];
			delete[] dataIQThreads[t];
		}

		break;
	}
	case DATA_TYPE::STRETCH: {
		int dataAD_size = (frame_len - 256) / 2;
		int16_t* dataAD = new int16_t[dataAD_size];

		int file_idx = 0;  // file need to read

		for (int i = 0; i < paras.echo_num; ++i) {
			file_idx = dataWFileSn[i] / frame_num;

			ifs_vec[file_idx].seekg((dataWFileSn[i] - file_idx * frame_num) * frame_len + 256, std::ifstream::beg);
			ifs_vec[file_idx].read((char*)dataAD, dataAD_size * sizeof(int16_t));

			for (int j = 0; (j + 1) < dataAD_size; j += 2) {
				h_data[i * (dataAD_size / 2) + (j / 2)] = std::complex<float>(static_cast<float>(dataAD[j]), static_cast<float>(dataAD[j + 1]));
			}
		}

		// d_data (host -> device)
		checkCudaErrors(cudaMemcpy(d_data, h_data, sizeof(cuComplex) * paras.data_num, cudaMemcpyHostToDevice));

		break;

		// multi-thread version
		// [todo] slower than single thread version ???
		/*// Split the workload among CPU threads
		int numThreads = 8;
		int subsetSize = paras.echo_num / numThreads;

		// Create a vector to hold the CPU threads
		std::vector<std::thread> threads(numThreads);

		// Create dataAD arrays for each thread to avoid data races
		int dataAD_size = (frame_len - 256) / 2;

		std::vector<int16_t*> dataADThreads(numThreads);
		for (int t = 0; t < numThreads; ++t) {
			dataADThreads[t] = new int16_t[dataAD_size];
		}

		// Launch the CPU threads
		for (int t = 0; t < numThreads; ++t) {
			int startIdx = t * subsetSize;
			int endIdx = (t == numThreads - 1) ? paras.echo_num : (startIdx + subsetSize);

			threads[t] = std::thread(multiThreadSTRETCH, h_data, d_data, dataADThreads[t], std::ref(ifs_vec), \
				startIdx, endIdx, dataAD_size, frame_num, frame_len, \
				std::ref(paras), std::ref(dataWFileSn), std::ref(dataNOut));
		}

		// Wait for all the threads to finish
		for (int t = 0; t < numThreads; ++t) {
			threads[t].join();
		}

		// Clean up the dynamically allocated arrays
		for (int t = 0; t < numThreads; ++t) {
			delete[] dataADThreads[t];
		}

		break;*/
	}
	default:
		break;
	}

	// Transfer Velocity Data(host to device)
	checkCudaErrors(cudaMemcpy(d_velocity, dataNOut.data() + paras.echo_num, sizeof(double) * paras.echo_num, cudaMemcpyHostToDevice));

	// close file
	for (int i = 0; i < m_file_vec.size(); ++i) {
		ifs_vec[i].close();
	}

	return EXIT_SUCCESS;
}


int ioOperation::uniformSampling(vec1D_INT* dataWFileSn, vec1D_DBL* dataNOut, vec1D_FLT* turnAngleOut, \
	const vec1D_DBL& dataN, const vec1D_FLT& turnAngle, const int& frame_num, const int& sampling_stride, const int& window_head, const int& window_len)
{
	// dataWFileSn = window_head:sampling_stride:window_end;
	for (int i = 0; i < window_len; ++i) {
		dataWFileSn->at(i) = window_head + i * sampling_stride;
	}

	int frame_num_total = frame_num * static_cast<int>(m_file_vec.size());

	// extracting dataNout from dataN based on index from dataWFileSn
	std::transform(dataWFileSn->cbegin(), dataWFileSn->cend(), dataNOut->begin() + 0 * window_len, [&](const int& x) {return dataN[x + 0 * frame_num_total]; });
	std::transform(dataWFileSn->cbegin(), dataWFileSn->cend(), dataNOut->begin() + 1 * window_len, [&](const int& x) {return dataN[x + 1 * frame_num_total]; });
	std::transform(dataWFileSn->cbegin(), dataWFileSn->cend(), dataNOut->begin() + 2 * window_len, [&](const int& x) {return dataN[x + 2 * frame_num_total]; });
	std::transform(dataWFileSn->cbegin(), dataWFileSn->cend(), dataNOut->begin() + 3 * window_len, [&](const int& x) {return dataN[x + 3 * frame_num_total]; });

	// extracting turnANgleOut from turnANgle based on index from dataWFileSn
	std::transform(dataWFileSn->cbegin(), dataWFileSn->cend(), turnAngleOut->begin(), [&](const int& x) {return std::abs(turnAngle[x]); });

	return EXIT_SUCCESS;
}


int ioOperation::nonUniformSampling() {
	// [todo] Implementing non-uniform sampling
	return EXIT_SUCCESS;
}


int ioOperation::writeFile(const std::string& outFilePath, const cuComplex* data, const  size_t& data_size)
{
	std::ofstream ofs(outFilePath);
	if (!ofs.is_open()) {
		std::cout << "[writeFile/WARN] Cannot open the file: " << outFilePath << std::endl;
		return EXIT_FAILURE;
	}

	for (int idx = 0; idx < data_size; idx++) {
		ofs << std::fixed << std::setprecision(5) << data[idx].x << "\n" << data[idx].y << "\n";
	}

	ofs.close();
	return EXIT_SUCCESS;
}


int ioOperation::writeFile(const std::string& outFilePath, const cuDoubleComplex* data, const  size_t& data_size)
{
	std::ofstream ofs(outFilePath);
	if (!ofs.is_open()) {
		std::cout << "[writeFile/WARN] Cannot open the file: " << outFilePath << std::endl;
		return EXIT_FAILURE;
	}

	for (int idx = 0; idx < data_size; idx++) {
		ofs << std::fixed << std::setprecision(5) << data[idx].x << "\n" << data[idx].y << "\n";
	}

	ofs.close();
	return EXIT_SUCCESS;
}


int ioOperation::writeFile(const std::string& outFilePath, const float* data, const  size_t& data_size)
{
	std::ofstream ofs(outFilePath);
	if (!ofs.is_open()) {
		std::cout << "[writeFile/WARN] Cannot open the file: " << outFilePath << std::endl;
		return EXIT_FAILURE;
	}

	for (int idx = 0; idx < data_size; idx++) {
		ofs << std::fixed << std::setprecision(5) << data[idx] << "\n";
	}

	ofs.close();
	return EXIT_SUCCESS;
}


int ioOperation::writeFile(const std::string& outFilePath, const double* data, const  size_t& data_size)
{
	std::ofstream ofs(outFilePath);
	if (!ofs.is_open()) {
		std::cout << "[writeFile/WARN] Cannot open the file: " << outFilePath << std::endl;
		return EXIT_FAILURE;
	}

	for (int idx = 0; idx < data_size; idx++) {
		ofs << std::fixed << std::setprecision(5) << data[idx] << "\n";
	}

	ofs.close();
	return EXIT_SUCCESS;
}
