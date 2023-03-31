#include "common.cuh"


CUDAHandle::CUDAHandle(const int& echo_num, const int& range_num)
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

CUDAHandle::~CUDAHandle()
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


void vecMulvec(cublasHandle_t handle, cuComplex* result_matrix, thrust::device_vector<comThr>& vec1, thrust::device_vector<comThr>& vec2, const cuComplex& alpha)
{
	int vec1_len = static_cast<int>(vec1.size());
	int vec2_len = static_cast<int>(vec2.size());

	cuComplex* d_vec1 = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(vec1.data()));
	cuComplex* d_vec2 = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(vec2.data()));

	checkCudaErrors(cublasCgeru(handle, vec1_len, vec2_len, &alpha, d_vec1, 1, d_vec2, 1, result_matrix, vec1_len));
}


void vecMulvec(cublasHandle_t handle, cuComplex* d_vec1, int len1, cuComplex* d_vec2, int len2, cuComplex* d_res_matrix, const cuComplex& alpha)
{
	checkCudaErrors(cublasCgeru(handle, len1, len2, &alpha, d_vec1, 1, d_vec2, 1, d_res_matrix, len1));
}


void vecMulvec(cublasHandle_t handle, float* result_matrix, thrust::device_vector<float>& vec1, thrust::device_vector<float>& vec2, const float& alpha)
{
	int vec1_len = static_cast<int>(vec1.size());
	int vec2_len = static_cast<int>(vec2.size());

	float* d_vec1 = reinterpret_cast<float*>(thrust::raw_pointer_cast(vec1.data()));
	float* d_vec2 = reinterpret_cast<float*>(thrust::raw_pointer_cast(vec2.data()));

	checkCudaErrors(cublasSger(handle, vec1_len, vec2_len, &alpha, d_vec1, 1, d_vec2, 1, result_matrix, vec1_len));
}

void vecMulvec(cublasHandle_t handle, float* d_vec1, int len1, float* d_vec2, int len2, float* d_res_matrix, const float& alpha)
{
	checkCudaErrors(cublasSger(handle, len1, len2, &alpha, d_vec1, 1, d_vec2, 1, d_res_matrix, len1));
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


__global__ void elementwiseMean(cuComplex* a, cuComplex* b, cuComplex* c, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		c[tid] = make_cuComplex((cuCrealf(a[tid]) + cuCrealf(b[tid])) / 2, (cuCimagf(a[tid]) + cuCimagf(b[tid])) / 2);
	}
}


__global__ void elementwiseMultiply(cuComplex* a, cuComplex* b, cuComplex* c, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		c[tid] = cuCmulf(a[tid], b[tid]);
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


__global__ void elementwiseMultiply(float* a, float* b, float* c, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		c[tid] = a[tid] * b[tid];
	}
}


/// <summary>
/// c = b ./ repmat(a, (len_b/len_a), 1)
/// </summary>
/// <param name="a"> len_a </param>
/// <param name="b"> len_b </param>
/// <param name="c"> len_c == len_b </param>
/// <param name="len_a"></param>
/// <param name="len_b"></param>
/// <returns></returns>
__global__ void elementwiseDivRep(float* a, float* b, float* c, int len_a, int len_b)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len_b) {
		c[tid] = b[tid] / a[tid % len_a];
	}
}


__global__ void elementwiseMultiply(float* a, cuComplex* b, cuComplex* c, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		c[tid] = cuCmulf(make_cuComplex(a[tid], 0.0f), b[tid]);
	}
}


__global__ void elementwiseMultiplyRep(float* a, cuComplex* b, cuComplex* c, int len_a, int len_b)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len_b) {
		c[tid] = cuCmulf(make_cuComplex(a[tid % len_a], 0.0f), b[tid]);
	}
}


__global__ void elementwiseMultiply(float* a, cuComplex* b, float* c, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		c[tid] = a[tid] * cuCabsf(b[tid]);
	}
}


__global__ void expJ(float* x, cuComplex* res, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len) {
		res[tid] = make_cuComplex(std::cos(x[tid]), std::sin(x[tid]));
	}
}


__global__ void genHammingVec(float* hamming, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < (len / 2)) {
		int tx = tid;
		hamming[tid] = (0.54f - 0.46f * std::cos(2 * PI_h * (static_cast<float>(tx) / len - 1)));
	}
	else if (tid < len) {
		int tx = len - tid - 1;
		hamming[tid] = (0.54f - 0.46f * std::cos(2 * PI_h * (static_cast<float>(tx) / len - 1)));
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


//void getMaxInColumns(thrust::device_vector<float>& c, thrust::device_vector<float>& maxval, thrust::device_vector<int>& maxidx, int row, int col)
//{
//	thrust::reduce_by_key(
//		thrust::make_transform_iterator(
//			thrust::make_counting_iterator((int)0),
//			thrust::placeholders::_1 / row),
//		thrust::make_transform_iterator(
//			thrust::make_counting_iterator((int)0),
//			thrust::placeholders::_1 / row) + row * col,
//		thrust::make_zip_iterator(
//			thrust::make_tuple(
//				thrust::make_permutation_iterator(
//					c.begin(),
//					thrust::make_transform_iterator(
//						thrust::make_counting_iterator((int)0), (thrust::placeholders::_1 % row) * col + thrust::placeholders::_1 / row)),
//				thrust::make_transform_iterator(
//					thrust::make_counting_iterator((int)0), thrust::placeholders::_1 % row))),
//		thrust::make_discard_iterator(),
//		thrust::make_zip_iterator(
//			thrust::make_tuple(
//				maxval.begin(),
//				maxidx.begin())),
//		thrust::equal_to<int>(),
//		thrust::maximum<thrust::tuple<float, int> >()
//	);
//}


__global__ void maxCols(float* d_data, float* d_max_clos, int rows, int cols)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int nTPB = blockDim.x;

	// [todo] Possible optimization:  halve the number of threads and size of shared memory assigned for each block.
	// Perform a reduction within the block to compute the final maximum value.
	// sdata_max_cols_idx store the index of the maximum value in each block.
	extern __shared__ int sdata_max_cols_idx[];
	sdata_max_cols_idx[tid] = tid * cols + bid;
	__syncthreads();

	for (int s = (nTPB >> 1); s > 0; s >>= 1) {
		if (tid < s) {
			if (d_data[sdata_max_cols_idx[tid]] < d_data[sdata_max_cols_idx[tid + s]]) {
				sdata_max_cols_idx[tid] = sdata_max_cols_idx[tid + s];
			}
		}
		__syncthreads();
	}

	if (tid == 0) {
		d_max_clos[bid] = d_data[sdata_max_cols_idx[0]];
	}
}


//template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
//OutputIterator expand(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator output)
//{
//	typedef typename thrust::iterator_difference<InputIterator1>::type difference_type;
//
//	difference_type input_size = thrust::distance(first1, last1);
//	difference_type output_size = thrust::reduce(first1, last1);
//
//	// scan the counts to obtain output offsets for each input element
//	thrust::device_vector<difference_type> output_offsets(input_size, 0);
//	thrust::exclusive_scan(first1, last1, output_offsets.begin());
//
//	// scatter the nonzero counts into their corresponding output positions
//	thrust::device_vector<difference_type> output_indices(output_size, 0);
//	thrust::scatter_if(thrust::counting_iterator<difference_type>(0), thrust::counting_iterator<difference_type>(input_size), output_offsets.begin(), first1, output_indices.begin());
//
//	// compute max-scan over the output indices, filling in the holes
//	thrust::inclusive_scan(output_indices.begin(), output_indices.end(), output_indices.begin(), thrust::maximum<difference_type>());
//
//	// gather input values according to index array (output = first2[output_indices])
//	OutputIterator output_end = output; thrust::advance(output_end, output_size);
//	thrust::gather(output_indices.begin(), output_indices.end(), first2, output);
//
//	// return output + output_size
//	thrust::advance(output, output_size);
//	return output;
//}
//
//
//void sumRows_thr(cuComplex* d_data, cuComplex* d_sum_rows, const int& row, const int& col)
//{
//	if (row == 1) {
//		if (d_data != d_sum_rows) {
//			checkCudaErrors(cudaMemcpy(d_sum_rows, d_data, sizeof(cuComplex) * col, cudaMemcpyDeviceToDevice));
//		}
//		return;
//	}
//
//	thrust::device_ptr<comThr> thr_d_data = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_data));
//	thrust::device_ptr<comThr> thr_d_sum_rows = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_sum_rows));
//
//	// * Generating keys for reduce_by_key
//	thrust::device_vector<int> oriKey(row);
//	thrust::sequence(thrust::device, oriKey.begin(), oriKey.end(), 1);
//
//	thrust::device_vector<int> d_counts(row, col);
//	thrust::device_vector<int> keyVec(row * col);  // 1,...,1,2,...,2,...,row,...,row
//
//	// * Expand keys according to counts
//	expand(d_counts.begin(), d_counts.end(), oriKey.begin(), keyVec.begin());
//
//	// * Sum mulRes in rows
//	thrust::reduce_by_key(thrust::device, keyVec.begin(), keyVec.end(), thr_d_data, thrust::make_discard_iterator(), thr_d_sum_rows);
//}


__global__ void sumCols(float* d_data, float* d_sum_clos, int rows, int cols)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int nTPB = blockDim.x;

	// [todo] Possible optimization:  halve the number of threads and size of shared memory assigned for each block.
	// Perform a reduction within the block to compute the final sum
	extern __shared__ float sdata[];
	sdata[tid] = d_data[tid * cols + bid];
	__syncthreads();

	for (int s = (nTPB >> 1); s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		d_sum_clos[bid] = sdata[0];
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
	extern __shared__ cuComplex s_data[];
	s_data[tid] = t_sum;
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			s_data[tid] = cuCaddf(s_data[tid], s_data[tid + s]);
		}
		__syncthreads();
	}

	if (tid == 0) {
		d_sum_rows[bid] = s_data[0];
	}
}


void cutRangeProfile(cuComplex*& d_data, RadarParameters& paras, const int& range_num_cut, const CUDAHandle& handles)
{
	int data_num_cut = paras.echo_num * range_num_cut;

	dim3 block(256);  // block size
	dim3 grid((data_num_cut + block.x - 1) / block.x);  // grid size

	cuComplex* d_data_cut = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_data_cut, sizeof(cuComplex) * data_num_cut));

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

	// point d_data to newly allocated memory block, updating values of paras
	checkCudaErrors(cudaFree(d_data));
	d_data = d_data_cut;
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


void getHRRP(cuComplex* d_hrrp, cuComplex* d_data, float* hamming, const RadarParameters& paras, const CUDAHandle& handles)
{
	int swap_len = paras.data_num / 2;

	dim3 block(256);  // block size
	dim3 grid((paras.data_num + block.x - 1) / block.x);  // grid size
	dim3 grid_swap((swap_len + block.x - 1) / block.x);

	// d_data = d_data .* repmat(hamming, echo_num, 1)
	elementwiseMultiplyRep << <grid, block >> > (hamming, d_data, d_data, paras.range_num, paras.data_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// d_hrrp = fftshift(fft(d_data))
	checkCudaErrors(cufftExecC2C(handles.plan_all_echo_c2c, d_data, d_hrrp, CUFFT_FORWARD));
	swap_range<cuComplex> << <grid_swap, block >> > (d_hrrp, d_hrrp + swap_len, swap_len);  // fftshift
	checkCudaErrors(cudaDeviceSynchronize());
}


float getTurnAngle(const float& azimuth1, const float& pitching1, const float& azimuth2, const float& pitching2) {
	std::vector<float> vec_1({ std::sin(pitching1 / 180 * PI_h), \
		std::cos(pitching1 / 180 * PI_h) * std::cos(azimuth1 / 180 * PI_h), \
		std::cos(pitching1 / 180 * PI_h) * std::sin(azimuth1 / 180 * PI_h) });

	std::vector<float> vec_2({ std::sin(pitching2 / 180 * PI_h), \
		std::cos(pitching2 / 180 * PI_h) * std::cos(azimuth2 / 180 * PI_h), \
		std::cos(pitching2 / 180 * PI_h) * std::sin(azimuth2 / 180 * PI_h) });

	float ret = std::acos(vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1] + vec_1[2] * vec_2[2]) / PI_h * 180;

	return ret;
}

int turnAngleLine(std::vector<float>* turnAngle, const std::vector<float>& azimuth, const std::vector<float>& pitching) {

	std::vector<int> idx;
	int pitchingSize = static_cast<int>(pitching.size());
	for (int i = 0; i < pitchingSize - 1; ++i) {
		if (std::abs(pitching[i + 1] - pitching[i]) > 0.2) {
			idx.push_back(i);
		}
	}

	std::vector<int> blkBeginNum;
	std::vector<int> blkEndNum;
	std::vector<int> blkLen;
	int idxSize = idx.empty() ? 1 : static_cast<int>(idx.size());

	blkBeginNum.insert(blkBeginNum.cend(), -1);
	blkBeginNum.insert(blkBeginNum.cend(), idx.begin(), idx.end());
	int blkSize = static_cast<int>(blkBeginNum.size());
	std::for_each(blkBeginNum.begin(), blkBeginNum.end(), [](int& x) {x++; });  // todo: add parallel execution policy


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
			std::vector<int> x = [=]() {
				std::vector<int> v;
				for (int i = 0; (i + stride) <= N; i += stride) {
					v.push_back(i);
				}
				return v;
			}();  // todo: range generate
			std::vector<float> Y = [=]() {
				std::vector<float> v;
				int xSize = static_cast<int>(x.size());
				for (int i = 0; i < xSize; ++i) {  // interpolation movement
					v.push_back(turnAngle->at(x[i]));
				}
				return v;
			}();
			std::vector<float> turnAngleInterp = [=]() {
				std::vector<float> v;
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


float interpolate(const std::vector<int>& xData, const std::vector<float>& yData, const int& x, const bool& extrapolate) {
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

int uniformSamplingFun(int* flagDataEnd, std::vector<int>* dataWFileSn, vec2D_FLOAT* dataNOut, std::vector<float>* turnAngleOut, \
	const vec2D_FLOAT& dataN, const std::vector<float>& turnAngle, const int& sampling_stride, const int& window_head, const int& window_len)
{
	int window_end = window_head + sampling_stride * window_len - 1;
	if (window_end > turnAngle.size()) {
		// flag_data_end = 1;
		// DataW_FileSn = [];
		// TurnAngleOut = [];
		// DataNOut = [];
		// return;
	}

	// 	DataW_FileSn = window_head:sampling_stride:window_end;
	*dataWFileSn = [=]() {  // todo: range generate
		std::vector<int> v;
		for (int i = window_head; (i + sampling_stride) <= window_end + 1; ++i) {
			v.push_back(i);
		}
		return v;
	}();

	// TurnAngleOut = abs(TurnAngle(window_head:sampling_stride:window_end));
	turnAngleOut->assign(dataWFileSn->size(), 0);
	std::transform(dataWFileSn->cbegin(), dataWFileSn->cend(), turnAngleOut->begin(), [=](int x) {return std::abs(turnAngle[x]); });

	// DataNOut = DataN(window_head:sampling_stride:window_end, : );
	dataNOut->assign(dataWFileSn->size(), std::vector<float>(8, 0));
	std::transform(dataWFileSn->cbegin(), dataWFileSn->cend(), dataNOut->begin(), [=](int x) {return dataN[x]; });
	*flagDataEnd = 0;

	return EXIT_SUCCESS;
}

int nonUniformSamplingFun() {
	return EXIT_SUCCESS;
}


/* ioOperation Class */
ioOperation::ioOperation(const std::string& dirPath, const int& fileType) :
	m_dirPath(dirPath), m_fileType(fileType)
{
	fs::directory_entry fsDirPath(m_dirPath);
	if (fsDirPath.is_directory() == false) {
		std::cout << "Invalid directory name!\n";
		return;
	}

	const std::vector<std::string> FILE_TYPE = { "00_1100.wbd" , "00_1101.wbd" };
	for (const auto& it : fs::directory_iterator{ fsDirPath }) {
		std::string fileStr = it.path().string();

		if (fileStr.substr(fileStr.length() - 11) == FILE_TYPE[fileType]) {
			m_filePath = fileStr;
			std::cout << "---* " << m_filePath << " *---\n\n";
			return;
		}
	}
}


ioOperation::~ioOperation() {}


int ioOperation::getSystemParasFirstFileStretch(RadarParameters* paras, int* frame_len, int* frame_num)
{
	std::ifstream ifs;
	ifs.open(m_filePath, std::ios_base::in | std::ios_base::binary);
	if (!ifs) {
		std::cout << "Cannot open file " << m_filePath << " !\n";
		return EXIT_FAILURE;
	}

	ifs.seekg(0, ifs.beg);

	uint32_t temp[36]{};
	ifs.read((char*)&temp, sizeof(temp));

	*frame_len = static_cast<int>(temp[4] * 4);  // length of frame, including frame head and orthogonal demodulation data.(unit: Byte)
	paras->fc = static_cast<long long>(temp[12] * 1e6);  // signal carrier frequency
	paras->band_width = static_cast<long long>(temp[13] * 1e6);  // signal band width
	//float PRI = temp[14] / 1e6;  // pulse repetition interval
	//float PRF = 1 / PRI;
	paras->Tp = static_cast<float>(temp[15] / 1e6);  // pulse width
	paras->Fs = static_cast<int>((temp[17] % static_cast<int>(std::pow(2, 16))) * 1e6);  // sampling frequency
	*frame_num = static_cast<int>(fs::file_size(fs::path(m_filePath))) / *frame_len;

	ifs.close();

	return EXIT_SUCCESS;
}

int ioOperation::readKuIFDSALLNBStretch(vec2D_FLOAT* dataN, vec2D_INT* stretchIndex, std::vector<float>* turnAngle, int* pulse_num_all, \
	const RadarParameters& paras, const int& frame_len, const int& frame_num)
{
	std::ifstream ifs;
	ifs.open(m_filePath, std::ios_base::in | std::ios_base::binary);
	if (!ifs) {
		std::cout << "Cannot open file " << m_filePath << " !\n";
		return EXIT_FAILURE;
	}

	dataN->assign(frame_num, std::vector<float>(8, 0));
	stretchIndex->assign(frame_num, std::vector<int>(2, 0));

	std::vector<float> azimuthVec(frame_num, 0);
	std::vector<float> pitchingVec(frame_num, 0);

	float timeYear = 0;  // time info only need read once
	float timeMonth = 0;
	float timeDay = 0;
	for (int i = 0; i < frame_num; i++) {
		ifs.seekg(i * frame_len + 40, ifs.beg);

		stretchIndex->at(i) = std::vector<int>({ i * frame_len + 256, frame_len });

		uint64_t sysTime = 0;
		ifs.read((char*)&sysTime, sizeof(uint64_t));

		uint32_t headerData[11]{};
		ifs.read((char*)&headerData, sizeof(headerData));

		float range = static_cast<float>(headerData[7]) * 0.1f;  // unit: m
		float velocity = static_cast<float>(headerData[8]);  // unit: m/s
		float azimuth = static_cast<float>(headerData[9]);
		float pitching = static_cast<float>(headerData[10]);

		/*if (velocity > std::pow(2, 31)) {
			velocity = (velocity - std::pow(2, 32)) * 0.1;
		}
		else {
			velocity = velocity * 0.1;
		}*/
		velocity = (velocity - (velocity > static_cast<float>(std::pow(2, 31)) ? static_cast<float>(std::pow(2, 32)) : 0)) * 0.1f;


		/*if (azimuth > std::pow(2, 31)) {
			azimuth = (azimuth - std::pow(2, 32)) * (360 / std::pow(2, 24));
		}
		else {
			azimuth = azimuth * (360 / std::pow(2, 24));
		}*/
		/*if (azimuth < 0) {
			azimuth = azimuth + 360;
		}*/
		azimuth = (azimuth - (azimuth > static_cast<float>(std::pow(2, 31)) ? static_cast<float>(std::pow(2, 32)) : 0)) * (360 / static_cast<float>(std::pow(2, 24)));
		azimuth += (azimuth < 0 ? 360 : 0);

		/*if (pitching > std::pow(2, 31)) {
			pitching = (pitching - std::pow(2, 32)) * (360 / std::pow(2, 24));
		}
		else {
			pitching = pitching * (360 / std::pow(2, 24));
		}*/
		/*if (pitching < 0) {
			pitching = pitching + 360;
		}*/
		pitching = (pitching - (pitching > static_cast<float>(std::pow(2, 31)) ? static_cast<float>(std::pow(2, 32)) : 0)) * (360 / static_cast<float>(std::pow(2, 24)));
		pitching += (pitching < 0 ? 360 : 0);

		ifs.seekg(i * frame_len + 32, ifs.beg);

		if (i == 0) {
			ifs.read((char*)&timeYear, sizeof(uint16_t));
			ifs.read((char*)&timeMonth, sizeof(uint8_t));
			ifs.read((char*)&timeDay, sizeof(uint8_t));
		}
		dataN->at(i) = std::vector<float>({ range, velocity, azimuth, pitching, static_cast<float>(sysTime), timeYear, timeMonth, timeDay });
		azimuthVec[i] = azimuth;
		pitchingVec[i] = pitching;
	}

	turnAngleLine(turnAngle, azimuthVec, pitchingVec);
	*pulse_num_all = static_cast<int>(turnAngle->size());

	ifs.close();

	return EXIT_SUCCESS;
}

int ioOperation::getKuDatafileSn(int* flagDataEnd, std::vector<int>* dataWFileSn, vec2D_FLOAT* dataNOut, std::vector<float>* turnAngleOut, \
	const vec2D_FLOAT& dataN, const RadarParameters& paras, const std::vector<float>& turnAngle, const int& sampling_stride, const int& window_head, const int& window_len, const bool& nonUniformSampling)
{

	if (nonUniformSampling == true) {
		nonUniformSamplingFun();
	}
	else {
		uniformSamplingFun(flagDataEnd, dataWFileSn, dataNOut, turnAngleOut, dataN, turnAngle, sampling_stride, window_head, window_len);
	}

	return EXIT_SUCCESS;
}

int ioOperation::getKuDataStretch(vec1D_COM_FLOAT* dataW, std::vector<int>* frameHeader, \
	const vec2D_INT& stretchIndex, const std::vector<int>& dataWFileSn)
{
	std::ifstream ifs;
	ifs.open(m_filePath, std::ios_base::in | std::ios_base::binary);
	if (!ifs) {
		std::cout << "Cannot open file " << m_filePath << " !\n";
		return EXIT_FAILURE;
	}

	int dataWFileSnSize = static_cast<int>(dataWFileSn.size());  // row of dataW
	for (int i = 0; i < dataWFileSnSize; ++i) {
		//fseek(fid1, StretchIndex(DataW_FileSn(i), 1), 'bof');
		ifs.seekg(stretchIndex[dataWFileSn[i]][0], ifs.beg);

		//DataAD = fread(fid1, (StretchIndex(DataW_FileSn(i), 2) - 256) / 2, 'int16');
		int dataADTempSize = (stretchIndex[dataWFileSn[i]][1] - 256) / 2;  // todo: frame_len???
		int16_t* dataADTemp = new int16_t[dataADTempSize];
		ifs.read((char*)dataADTemp, dataADTempSize * sizeof(int16_t));


		if (i == 0) {
			dataW->resize(dataWFileSnSize * (dataADTempSize / 2));
		}

		//data_AD = DataAD(1:2 : end) + 1i * DataAD(2:2 : end);
		//DataW(i, :) = data_AD.';
		for (int j = 0; (j + 1) < dataADTempSize; j += 2) {
			dataW->at(i * (dataADTempSize / 2) + (j / 2)) = std::complex<float>(static_cast<float>(dataADTemp[j]), static_cast<float>(dataADTemp[j + 1]));
		}

		delete[] dataADTemp;
		dataADTemp = nullptr;
	}

	/*
	fseek(fid1, StretchIndex(DataW_FileSn(1), 1) - 256, 'bof');
	DataRead = fread(fid1, 108, 'uint8');
	FrameHeader = [DataRead(1:12, 1); DataRead(101:104, 1); DataRead(77:92, 1); DataRead(97:100, 1); DataRead(33:38, 1); DataRead(31, 1); DataRead(105:108, 1); DataRead(61:64, 1); ];
	*/
	ifs.seekg(stretchIndex[dataWFileSn[0]][0] - 256, ifs.beg);

	uint8_t frameHeaderTemp[108]{};
	ifs.read((char*)&frameHeaderTemp, sizeof(frameHeaderTemp));

	frameHeader->insert(frameHeader->cend(), frameHeaderTemp + 0, frameHeaderTemp + 12);
	frameHeader->insert(frameHeader->cend(), frameHeaderTemp + 100, frameHeaderTemp + 104);
	frameHeader->insert(frameHeader->cend(), frameHeaderTemp + 76, frameHeaderTemp + 92);
	frameHeader->insert(frameHeader->cend(), frameHeaderTemp + 96, frameHeaderTemp + 100);
	frameHeader->insert(frameHeader->cend(), frameHeaderTemp + 32, frameHeaderTemp + 38);
	frameHeader->insert(frameHeader->cend(), frameHeaderTemp + 30, frameHeaderTemp + 31);
	frameHeader->insert(frameHeader->cend(), frameHeaderTemp + 104, frameHeaderTemp + 108);
	frameHeader->insert(frameHeader->cend(), frameHeaderTemp + 60, frameHeaderTemp + 64);

	return EXIT_SUCCESS;
}


int ioOperation::writeFile(const std::string& outFilePath, const std::complex<float>* data, const  size_t& data_size)
{
	std::ofstream ofs(outFilePath);
	if (!ofs.is_open()) {
		std::cout << "Cannot open the file\n" << std::endl;
		return EXIT_FAILURE;
	}

	for (int idx = 0; idx < data_size; idx++) {
		ofs << std::fixed << std::setprecision(3) << data[idx].real() << "\n" << data[idx].imag() << "\n";
	}

	ofs.close();
	return EXIT_SUCCESS;
}


int ioOperation::writeFile(const std::string& outFilePath, const float* data, const  size_t& data_size)
{
	std::ofstream ofs(outFilePath);
	if (!ofs.is_open()) {
		std::cout << "Cannot open the file\n" << std::endl;
		return EXIT_FAILURE;
	}

	for (int idx = 0; idx < data_size; idx++) {
		ofs << std::fixed << std::setprecision(3) << data[idx] << "\n";
	}

	ofs.close();
	return EXIT_SUCCESS;
}


int ioOperation::dataWriteBack(const std::string& outFilePath, const cuComplex* d_data, const  size_t& data_size)
{

	std::complex<float>* h_data = new std::complex<float>[data_size];
	checkCudaErrors(cudaMemcpy(h_data, d_data, sizeof(cuComplex) * data_size, cudaMemcpyDeviceToHost));  // data (device -> host)

	ioOperation::writeFile(outFilePath, h_data, data_size);

	delete[] h_data;
	h_data = nullptr;

	return EXIT_SUCCESS;
}


int ioOperation::dataWriteBack(const std::string& outFilePath, const float* d_data, const  size_t& data_size)
{

	float* h_data = new float[data_size];
	checkCudaErrors(cudaMemcpy(h_data, d_data, sizeof(float) * data_size, cudaMemcpyDeviceToHost));  // data (device -> host)

	ioOperation::writeFile(outFilePath, h_data, data_size);

	delete[] h_data;
	h_data = nullptr;

	return EXIT_SUCCESS;
}
