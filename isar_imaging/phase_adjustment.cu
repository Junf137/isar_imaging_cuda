#include "phase_adjustment.cuh"


void dopplerTracking(cuComplex* d_data_comp, cuComplex* d_phase, cuComplex* d_data, const int& echo_num, const int& range_num, const bool& if_compensation)
{
	int data_num = echo_num * range_num;
	int data_num_less_echo = (echo_num - 1) * range_num;

	dim3 block(DEFAULT_THREAD_PER_BLOCK);

	// * Applying conjugate multiplication on two successive rows
	cuComplex* d_mul_res = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_mul_res, sizeof(cuComplex) * data_num_less_echo));
	elementwiseMultiplyConjA << <(data_num_less_echo + block.x - 1) / block.x, block >> > (d_data, d_data + range_num, d_mul_res, data_num_less_echo);
	checkCudaErrors(cudaDeviceSynchronize());

	// * Sum d_mul_res in rows
	thrust::device_ptr<comThr> thr_phase = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_phase));
	sumRows << <echo_num - 1, block, block.x * sizeof(cuComplex) >> > (d_mul_res, d_phase + 1, echo_num - 1, range_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// * Getting compensation angle
	thrust::device_vector<float> angle(echo_num - 1);
	thrust::transform(thrust::device, thr_phase + 1, thr_phase + echo_num, angle.begin(), \
		[]__host__ __device__(const comThr & x) { return thrust::arg(x); });
	thrust::inclusive_scan(thrust::device, angle.begin(), angle.end(), angle.begin());

	// * Calculating phase using angle
	thr_phase[0] = comThr(0.0f, 0.0f);
	thrust::transform(thrust::device, angle.begin(), angle.end(), thr_phase + 1, \
		[]__host__ __device__(const float& x) { return thrust::exp(comThr(0.0, -x)); });

	// * Compensation
	if (if_compensation == true) {
		diagMulMat << <(data_num + block.x - 1) / block.x, block >> > (d_phase, d_data, d_data_comp, range_num, data_num);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	// * Free Allocated Space
	checkCudaErrors(cudaFree(d_mul_res));
}


//void dopplerTracking_2(cuComplex* d_data_comp, cuComplex* d_phase, cuComplex* d_data, const int& echo_num, const int& range_num, const bool& if_compensation)
//{
//	int data_num = echo_num * range_num;
//	int data_num_less_echo = (echo_num - 1) * range_num;
//
//	dim3 block(DEFAULT_THREAD_PER_BLOCK);
//
//	// * Applying conjugate multiplication on two successive rows
//	cuComplex* d_mul_res = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_mul_res, sizeof(cuComplex) * data_num_less_echo));
//	elementwiseMultiplyConjA << <(data_num_less_echo + block.x - 1) / block.x, block >> > (d_data, d_data + range_num, d_mul_res, data_num_less_echo);
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	cuComplex* d_xw_1 = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_xw_1, sizeof(cuComplex) * (echo_num - 1)));
//	sumRows << <echo_num - 1, block, block.x * sizeof(cuComplex) >> > (d_mul_res, d_xw_1, echo_num - 1, range_num);
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	float* d_mul_res_abs = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_mul_res_abs, sizeof(float) * data_num_less_echo));
//	elementwiseAbs << <(data_num_less_echo + block.x - 1) / block.x, block >> > (d_mul_res, d_mul_res_abs, data_num);
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	float* d_xw_2 = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_xw_2, sizeof(float)* (echo_num - 1)));
//	sumRows << <echo_num - 1, block, block.x * sizeof(float) >> > (d_mul_res_abs, d_xw_2, echo_num - 1, range_num);
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	elementwiseDiv << <((echo_num - 1) + block.x - 1) / block.x, block >> > (d_xw_2, d_xw_1, d_xw_1, echo_num - 1);
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	cuComplex tmp = make_cuComplex(0.0f, 0.0f);
//	checkCudaErrors(cudaMemcpy(d_phase, &tmp, sizeof(cuComplex) * 1, cudaMemcpyHostToDevice));
//
//	elementwiseConj << <((echo_num - 1) + block.x - 1) / block.x, block >> > (d_xw_1, d_phase + 1, echo_num - 1);
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	// * Compensation
//	if (if_compensation == true) {
//		diagMulMat << <(data_num + block.x - 1) / block.x, block >> > (d_phase, d_data, d_data_comp, range_num, data_num);
//		checkCudaErrors(cudaDeviceSynchronize());
//	}
//
//	// * Free Allocated Space
//	checkCudaErrors(cudaFree(d_mul_res));
//	checkCudaErrors(cudaFree(d_xw_1));
//	checkCudaErrors(cudaFree(d_mul_res_abs));
//	checkCudaErrors(cudaFree(d_xw_2));
//}


//void rangeVariantPhaseComp(cuComplex* d_data, double* h_azimuth, double* h_pitch, const RadarParameters& paras, const CUDAHandle& handles)
//{
//	// [todo] expanding data width to double
//	// transfer angle information to device, calculating central angle
//	int mid_index = paras.echo_num / 2;
//	float middle_azimuth = h_azimuth[mid_index];
//	float middle_pitch = h_pitch[mid_index];
//
//	float* d_azimuth = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_azimuth, sizeof(float) * paras.echo_num));
//	checkCudaErrors(cudaMemcpy(d_azimuth, h_azimuth, sizeof(float) * paras.echo_num, cudaMemcpyHostToDevice));
//
//	float* d_pitch = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_pitch, sizeof(float) * paras.echo_num));
//	checkCudaErrors(cudaMemcpy(d_pitch, h_pitch, sizeof(float) * paras.echo_num, cudaMemcpyHostToDevice));
//
//	float resolution = static_cast<float>(LIGHT_SPEED) / static_cast<float>(2 * paras.band_width);
//	float wave_length = static_cast<float>(LIGHT_SPEED) / static_cast<float>(paras.fc);
//
//	// assuming the rotation center is in the middle of the 1D image, generating range vector
//	thrust::device_vector<float> range(paras.range_num);
//	float* d_range = thrust::raw_pointer_cast(range.data());
//	thrust::sequence(thrust::device, range.begin(), range.end(), -float(paras.range_num) / 2.0);
//
//	thrust::transform(thrust::device, range.begin(), range.end(), range.begin(), \
//		[=]__host__ __device__(const float& x) { return x * 2.0f * resolution / wave_length; });
//
//	// calculating turning angle of each echo comparing to that of middle echo signal
//	thrust::device_vector<float> theta(paras.echo_num);
//	float* d_theta = thrust::raw_pointer_cast(theta.data());
//	thrust::device_ptr<float> thr_azimuth = thrust::device_pointer_cast(d_azimuth);
//	thrust::device_ptr<float> thr_pitch = thrust::device_pointer_cast(d_pitch);
//	float mid_x = sinf(middle_pitch / 180 * PI_FLT);
//	float mid_y = cosf(middle_pitch / 180 * PI_FLT) * cosf(middle_azimuth / 180 * PI_FLT);
//	float mid_z = cosf(middle_pitch / 180 * PI_FLT) * sinf(middle_azimuth / 180 * PI_FLT);
//
//	thrust::transform(thrust::device, thr_azimuth, thr_azimuth + paras.echo_num, thr_pitch, theta.begin(), \
//		[=]__host__ __device__(const float& cur_azi, const float& cur_pit)
//	{
//		float x = sinf(cur_pit / 180 * PI_FLT);
//		float y = cosf(cur_pit / 180 * PI_FLT) * cosf(cur_azi / 180 * PI_FLT);
//		float z = cosf(cur_pit / 180 * PI_FLT) * sinf(cur_azi / 180 * PI_FLT);
//
//		float angle = (x * mid_x + y * mid_y + z * mid_z);
//		angle = acosf(angle);
//		float angle2;
//		angle2 = powf(angle, 2.0);
//		return angle2;
//	});
//
//	// build compensation matrix and compensate range sequence
//	float* d_comp_mat = nullptr;
//	checkCudaErrors(cudaMalloc((void**)&d_comp_mat, sizeof(float) * paras.data_num));
//	checkCudaErrors(cudaMemset(d_comp_mat, 0, sizeof(float) * paras.data_num));
//
//	// 08-05-2020 modified
//	float alpha = 1.0f;
//	checkCudaErrors(cublasSger(handles.handle, paras.range_num, paras.echo_num, &alpha, d_range, 1, d_theta, 1, d_comp_mat, paras.range_num));
//
//	thrust::device_ptr<float> thr_comp_mat = thrust::device_pointer_cast(d_comp_mat);
//	thrust::device_ptr<comThr> thr_data = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_data));
//
//	thrust::transform(thrust::device, thr_comp_mat, thr_comp_mat + paras.data_num, thr_data, thr_data, \
//		[]__host__ __device__(const float& x, const comThr & y) { return y * thrust::exp(comThr(0.0, -2 * PI_FLT * x)); });
//
//	// Free Allocated GPU Memory
//	checkCudaErrors(cudaFree(d_azimuth));
//	checkCudaErrors(cudaFree(d_pitch));
//	checkCudaErrors(cudaFree(d_comp_mat));
//}


void fastEntropy(cuComplex* d_data, const int& echo_num, const int& range_num, const CUDAHandle& handles)
{
	int data_num = echo_num * range_num;

	dim3 block(DEFAULT_THREAD_PER_BLOCK);  // block size
	dim3 grid((data_num + block.x - 1) / block.x);  // grid size

	// * Pre-processing and pre-imaging
	// d_data_abs = abs(d_data)
	float* d_data_abs = g_d_data_num_flt_1;
	elementwiseAbs << <grid, block >> > (d_data, d_data_abs, data_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// max_value = max(abs(d_data),[],1);
	float* d_max_val = g_d_range_num_flt_1;
	thrust::device_ptr<float> thr_max_val = thrust::device_pointer_cast(d_max_val);
	maxCols << <range_num, block, block.x * sizeof(int) >> > (d_data_abs, d_max_val, echo_num, range_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// mean_val = mean(max_value);
	float mean_val = thrust::reduce(thrust::device, thr_max_val, thr_max_val + range_num, 0, thrust::plus<int>()) / static_cast<float>(range_num);
	float threshold = 1.48f * mean_val;

	// tgt_index = find(max_value > mean_val*1.48);
	thrust::device_vector<int> tgt_index(range_num);
	auto end = thrust::copy_if(thrust::make_counting_iterator(0), thrust::make_counting_iterator(range_num), thr_max_val, tgt_index.begin(), \
		[=]__host__ __device__(const float& x) { return (x > threshold); });
	int tgt_num = static_cast<int>(end - tgt_index.begin());
	//tgt_index.resize(tgt_num);

	// tmpData = droptrace(RetData_RA);
	cuComplex* d_data_comp = g_d_data_num_com_flt_1;
	cuComplex* d_phase_tmp = g_d_echo_num_com_flt_1;
	thrust::device_ptr<comThr> thr_phase_tmp = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_phase_tmp));
	dopplerTracking(d_data_comp, d_phase_tmp, d_data, echo_num, range_num, true);

	//if (tgt_num < 5) {
	//	checkCudaErrors(cudaMemcpy(d_data, d_data_comp, sizeof(cuComplex)* data_num, cudaMemcpyDeviceToDevice));

	//	// * Free allocated GPU memory and return
	//	checkCudaErrors(cudaFree(d_data_abs));
	//	checkCudaErrors(cudaFree(d_max_val));
	//	checkCudaErrors(cudaFree(d_phase_tmp));
	//	checkCudaErrors(cudaFree(d_data_comp));
	//	std::cout << "tgt_num < 5\n";
	//	return;
	//}

	// d_img = fft(tmpData, [], 1);
	cuComplex* d_img = g_d_data_num_cut_com_flt_1;
	thrust::device_ptr<comThr> thr_img = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_img));
	checkCudaErrors(cufftExecC2C(handles.plan_all_range_c2c, d_data_comp, d_img, CUFFT_FORWARD));

	int unit1_num = tgt_num;
	int unit2_num = nextPow2(tgt_num / 2);

	// sqr_img = abs(image2).^2;
	float* d_sqr_img = g_d_data_num_flt_2;
	thrust::device_ptr<float> thr_sqr_img = thrust::device_pointer_cast(d_sqr_img);
	thrust::transform(thrust::device, thr_img, thr_img + data_num, thr_sqr_img, \
		[]__host__ __device__(const comThr & x) { return powf(thrust::abs(x), 2); });

	// sqr_image_sum_col = sum(sqr_img);
	float* d_sqr_img_sum_col = g_d_range_num_cut_flt_1;
	thrust::device_ptr<float> thr_sqr_img_sum_col = thrust::device_pointer_cast(d_sqr_img_sum_col);
	sumCols << <range_num, echo_num, echo_num * sizeof(float) >> > (d_sqr_img, d_sqr_img_sum_col, echo_num, range_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// * Get entropy
	float* d_sqr_img_norm = g_d_data_num_cut_flt_1;
	thrust::device_ptr<float> thr_sqr_img_norm = thrust::device_pointer_cast(d_sqr_img_norm);
	elementwiseDivRep << <grid, block >> > (d_sqr_img_sum_col, d_sqr_img, d_sqr_img_norm, range_num, data_num);
	checkCudaErrors(cudaDeviceSynchronize());

	thrust::transform(thrust::device, thr_sqr_img_norm, thr_sqr_img_norm + data_num, thr_sqr_img_norm, \
		[]__host__ __device__(float& x) { return -(x * logf(x)); });

	float* d_entropy = g_d_range_num_cut_flt_2;
	sumCols << <range_num, echo_num, echo_num * sizeof(float) >> > (d_sqr_img_norm, d_entropy, echo_num, range_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// * Store the index of the first unit1_num largest values of sqr_image_sum_col in select_bin
	thrust::device_vector<int> select_bin(range_num);
	thrust::sequence(thrust::device, select_bin.begin(), select_bin.end(), 0);
	thrust::stable_sort_by_key(thrust::device, thr_sqr_img_sum_col, thr_sqr_img_sum_col + range_num, select_bin.begin(), thrust::greater<float>());
	select_bin.resize(unit1_num);

	// * Store the index of the first unit2_num smallest values of entropy in select_bin
	thrust::device_vector<float> select_entropy(unit1_num);
	for (int ii = 0; ii < unit1_num; ii++) {
		select_entropy[ii] = d_entropy[select_bin[ii]];
	}
	thrust::stable_sort_by_key(thrust::device, select_entropy.begin(), select_entropy.end(), select_bin.begin());

	select_bin.resize(unit2_num);
	int* d_select_bin = thrust::raw_pointer_cast(select_bin.data());

	thrust::sort(thrust::device, select_bin.begin(), select_bin.end());

	// * Rebuild echo using select_bin
	int data_num_unit2 = echo_num * unit2_num;
	float scale_ifft = 1.0f / echo_num;

	cuComplex* d_new_data = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_new_data, sizeof(cuComplex) * data_num_unit2));
	thrust::device_ptr<comThr> thr_new_data = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_new_data));
	if (unit2_num >= 1024) {
		std::cout << "[fastEntropy/WARN] unit2_num >= 1024, please double-check the data or optimize data layout of selectRangeBins()." << std::endl;
		system("pause");
		exit(EXIT_FAILURE);
	}
	selectRangeBins << <echo_num, unit2_num >> > (d_new_data, d_data, d_select_bin, echo_num, range_num, unit2_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// * Doppler phase tracking
	dopplerTracking(d_new_data, d_phase_tmp, d_new_data, echo_num, unit2_num, false);

	// * Minimum entropy searching
	cuComplex* d_new_data_tmp = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_new_data_tmp, sizeof(cuComplex)* data_num_unit2));
	thrust::device_ptr<comThr> thr_new_data_tmp = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_new_data_tmp));

	// configure fft handle (along columns)
	cufftHandle plan_all_unit2_c2c;
	int batch = unit2_num;
	int rank = 1;
	int n[1] = { echo_num };
	int inembed[] = { echo_num };
	int onembed[] = { echo_num };
	int istride = unit2_num;
	int ostride = unit2_num;
	int idist = 1;
	int odist = 1;
	checkCudaErrors(cufftPlanMany(&plan_all_unit2_c2c, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch));

	for (int i = 0; i < FAST_ENTROPY_ITERATION_NUM; ++i) {
		diagMulMat << <(unit2_num * echo_num + block.x - 1) / block.x, block >> > (d_phase_tmp, d_new_data, d_new_data_tmp, unit2_num, echo_num* unit2_num);
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cufftExecC2C(plan_all_unit2_c2c, d_new_data_tmp, d_new_data_tmp, CUFFT_FORWARD));

		thrust::transform(thrust::device, thr_new_data_tmp, thr_new_data_tmp + data_num_unit2, thr_new_data_tmp, \
			[]__host__ __device__(comThr& x) { return x * thrust::abs(x); });

		checkCudaErrors(cufftExecC2C(plan_all_unit2_c2c, d_new_data_tmp, d_new_data_tmp, CUFFT_INVERSE));
		checkCudaErrors(cublasCsscal(handles.handle, data_num_unit2, &scale_ifft, d_new_data_tmp, 1));

		thrust::transform(thrust::device, thr_new_data, thr_new_data + data_num_unit2, thr_new_data_tmp, thr_new_data_tmp, \
			[=]__host__ __device__(const comThr& x, const comThr& y) { return thrust::conj(x) * y; });

		sumRows << <echo_num, unit2_num, unit2_num * sizeof(cuComplex) >> > (d_new_data_tmp, d_phase_tmp, echo_num, unit2_num);
		checkCudaErrors(cudaDeviceSynchronize());

		thrust::transform(thrust::device, thr_phase_tmp, thr_phase_tmp + echo_num, thr_phase_tmp, \
			[]__host__ __device__(const comThr& x) { return (x / thrust::abs(x)); });
	}

	diagMulMat << <(data_num + block.x - 1) / block.x, block >> > (d_phase_tmp, d_data, d_data, range_num, data_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// * Free allocated memory
	checkCudaErrors(cudaFree(d_new_data));
	checkCudaErrors(cudaFree(d_new_data_tmp));

	checkCudaErrors(cufftDestroy(plan_all_unit2_c2c));
}


__global__ void selectRangeBins(cuComplex* d_new_data, cuComplex* d_data, int* select_bin, int echo_num, int range_num, int unit2_num)
{
	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + tid;

	if (idx < echo_num * unit2_num) {
		int row_idx = static_cast<int>(idx / unit2_num);
		int ori_idx = row_idx * range_num + select_bin[tid];

		d_new_data[idx] = d_data[ori_idx];
	}
}
