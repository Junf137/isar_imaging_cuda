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

	// [todo] Possible optimization:  halve the number of threads and size of shared memory assigned for each block.
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

	// [todo] Possible optimization:  halve the number of threads and size of shared memory assigned for each block.
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


__global__ void sumRows(cuComplex* d_data, cuComplex* d_sum_rows, int rows, int cols)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	cuComplex t_sum = make_cuComplex(0.0f, 0.0f);
	for (int i = tid; i < cols; i += blockDim.x) {
		t_sum = cuCaddf(t_sum, d_data[bid * cols + i]);
	}

	// Perform a reduction within the block to compute the final sum
	extern __shared__ cuComplex sdata_sum_rows_com_flt[];
	sdata_sum_rows_com_flt[tid] = t_sum;
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
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

	float t_sum = 0.0f;
	for (int i = tid; i < cols; i += blockDim.x) {
		t_sum = t_sum + d_data[bid * cols + i];
	}

	// Perform a reduction within the block to compute the final sum
	extern __shared__ float sdata_sum_rows_flt[];
	sdata_sum_rows_flt[tid] = t_sum;
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata_sum_rows_flt[tid] = sdata_sum_rows_flt[tid] + sdata_sum_rows_flt[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		d_sum_rows[bid] = sdata_sum_rows_flt[0];
	}
}


void cutRangeProfile(cuComplex* d_data_cut, cuComplex* d_data, RadarParameters& paras, const int& range_num_cut, const CUDAHandle& handles)
{
	int data_num_cut = paras.echo_num * range_num_cut;

	dim3 block(DEFAULT_THREAD_PER_BLOCK);  // block size
	dim3 grid((data_num_cut + block.x - 1) / block.x);  // grid size

	// max(abs(d_data(1,:)))
	int range_abs_max_idx = 0;
	cuComplex range_abs_max_val = make_cuComplex(0.0f, 0.0f);
	getMax(handles.handle, d_data, paras.range_num, &range_abs_max_idx, &range_abs_max_val);

	int offset_l = range_abs_max_idx - range_num_cut / 2;
	int offset_r = range_abs_max_idx + range_num_cut / 2;
	if (offset_l < 0 || offset_r >= paras.range_num) {
		std::cout << "[cutRangeProfile/WARN] Invalid range_num_cut! Probably too long.\n" << std::endl;
		system("pause");
		exit(EXIT_FAILURE);
	}

	cutRangeProfileHelper << <grid, block >> > (d_data, d_data_cut, data_num_cut, offset_l, range_num_cut, paras.range_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// updating values of paras
	paras.range_num = range_num_cut;
	paras.data_num = paras.echo_num * paras.range_num;
}


__global__ void cutRangeProfileHelper(cuComplex* d_in, cuComplex* d_out, int data_num_cut, int offset, int range_num_cut, int range_num)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < data_num_cut) {
		d_out[tid] = d_in[(tid / range_num_cut) * range_num + offset + tid % range_num_cut];
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


void getHRRP(cuComplex* d_hrrp, cuComplex* d_data, const RadarParameters& paras, const CUDAHandle& handles)
{
	dim3 block(DEFAULT_THREAD_PER_BLOCK);  // block size

	// fft
	checkCudaErrors(cufftExecC2C(handles.plan_all_echo_c2c, d_data, d_hrrp, CUFFT_FORWARD));
	// fftshift
	ifftshiftRows << <dim3(((paras.range_num / 2) + block.x - 1) / block.x, paras.echo_num), block >> > (d_hrrp, paras.range_num);
	checkCudaErrors(cudaDeviceSynchronize());
}


float getTurnAngle(const float& azimuth1, const float& pitching1, const float& azimuth2, const float& pitching2) {
	vec1D_FLT vec_1({ std::sin(pitching1 / 180 * PI_FLT), \
		std::cos(pitching1 / 180 * PI_FLT) * std::cos(azimuth1 / 180 * PI_FLT), \
		std::cos(pitching1 / 180 * PI_FLT) * std::sin(azimuth1 / 180 * PI_FLT) });

	vec1D_FLT vec_2({ std::sin(pitching2 / 180 * PI_FLT), \
		std::cos(pitching2 / 180 * PI_FLT) * std::cos(azimuth2 / 180 * PI_FLT), \
		std::cos(pitching2 / 180 * PI_FLT) * std::sin(azimuth2 / 180 * PI_FLT) });

	float ret = std::acos(vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1] + vec_1[2] * vec_2[2]) / PI_FLT * 180;

	return ret;
}

int turnAngleLine(vec1D_FLT* turnAngle, const vec1D_FLT& azimuth, const vec1D_FLT& pitching) {

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


float interpolate(const vec1D_INT& xData, const vec1D_FLT& yData, const int& x, const bool& extrapolate) {
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


int uniformSampling(vec1D_INT* dataWFileSn, vec1D_DBL* dataNOut, vec1D_FLT* turnAngleOut, \
	const vec1D_DBL& dataN, const vec1D_FLT& turnAngle, const int& frame_num, const int& sampling_stride, const int& window_head, const int& window_len)
{
	// dataWFileSn = window_head:sampling_stride:window_end;
	for (int i = 0; i < window_len; ++i) {
		dataWFileSn->at(i) = window_head + i * sampling_stride;
	}

	// extracting dataNout from dataN based on index from dataWFileSn
	std::transform(dataWFileSn->cbegin(), dataWFileSn->cend(), dataNOut->begin() + 0 * window_len, [&](const int& x) {return dataN[x + 0 * frame_num]; });
	std::transform(dataWFileSn->cbegin(), dataWFileSn->cend(), dataNOut->begin() + 1 * window_len, [&](const int& x) {return dataN[x + 1 * frame_num]; });
	std::transform(dataWFileSn->cbegin(), dataWFileSn->cend(), dataNOut->begin() + 2 * window_len, [&](const int& x) {return dataN[x + 2 * frame_num]; });
	std::transform(dataWFileSn->cbegin(), dataWFileSn->cend(), dataNOut->begin() + 3 * window_len, [&](const int& x) {return dataN[x + 3 * frame_num]; });

	// extracting turnANgleOut from turnANgle based on index from dataWFileSn
	std::transform(dataWFileSn->cbegin(), dataWFileSn->cend(), turnAngleOut->begin(), [&](const int& x) {return std::abs(turnAngle[x]); });

	return EXIT_SUCCESS;
}


int nonUniformSampling() {
	return EXIT_SUCCESS;
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

	vec1D_FLT azimuthVec(frame_num_total);  // todo: expanding to double?
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


int ioOperation::getKuData(vec1D_COM_FLT* dataW, \
	const int& frame_len, const vec1D_INT& dataWFileSn, const int& window_len)
{
	return EXIT_SUCCESS;
}


int ioOperation::getKuDataStretch(vec1D_COM_FLT* dataW, \
	const int& frame_len, const vec1D_INT& dataWFileSn, const int& window_len)
{
	std::ifstream ifs;
	ifs.open(m_file_vec[0], std::ios_base::in | std::ios_base::binary);
	if (!ifs) {
		std::cout << "[getKuDataStretch/WARN] Cannot open file " << m_file_vec[0] << " !\n";
		return EXIT_FAILURE;
	}

	int dataAD_size = (frame_len - 256) / 2;
	int16_t* dataAD = new int16_t[dataAD_size];
	
	for (int i = 0; i < window_len; ++i) {
		ifs.seekg(dataWFileSn[i] * frame_len + 256, std::ifstream::beg);

		ifs.read((char*)dataAD, dataAD_size * sizeof(int16_t));

		for (int j = 0; (j + 1) < dataAD_size; j += 2) {
			dataW->at(i * (dataAD_size / 2) + (j / 2)) = std::complex<float>(static_cast<float>(dataAD[j]), static_cast<float>(dataAD[j + 1]));
		}
	}

	delete[] dataAD;
	dataAD = nullptr;

	ifs.close();

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
