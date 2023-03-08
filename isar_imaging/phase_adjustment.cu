#include "phase_adjustment.cuh"


void dopplerTracking(cuComplex* d_data, const int& echo_num, const int& range_num)
{
	int data_num = echo_num * range_num;
	int data_num_less_echo = (echo_num - 1) * range_num;

	// * Generating keys for reduce_by_key
	thrust::device_vector<int> oriKey(echo_num - 1);
	thrust::sequence(thrust::device, oriKey.begin(), oriKey.end(), 1);

	thrust::device_vector<int> d_counts(echo_num - 1, range_num);
	thrust::device_vector<int> keyVec(data_num_less_echo);    // 1,...,1,...,...,255,...,255

	// * Expand keys according to counts															  
	expand(d_counts.begin(), d_counts.end(), oriKey.begin(), keyVec.begin());

	// * Applying conjugate multiplication on two successive raws
	thrust::device_ptr<comThr> thr_d_data = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_data));

	thrust::device_vector<comThr> mulRes(data_num_less_echo);
	thrust::transform(thrust::device, thr_d_data, thr_d_data + data_num_less_echo, thr_d_data + range_num, mulRes.begin(), \
		[]__host__ __device__(const comThr & x, const comThr & y) { return thrust::conj(x) * y; });

	// * Sum mulRes in raws
	thrust::device_vector<comThr> xw(echo_num - 1);
	thrust::reduce_by_key(thrust::device, keyVec.begin(), keyVec.end(), mulRes.begin(), thrust::make_discard_iterator(), xw.begin());

	// * Getting compensation andle
	thrust::device_vector<float> angle(echo_num - 1);
	thrust::transform(thrust::device, xw.begin(), xw.end(), angle.begin(), \
		[]__host__ __device__(const comThr& x) { return thrust::arg((x / thrust::abs(x))); });

	thrust::inclusive_scan(thrust::device, angle.begin(), angle.end(), angle.begin());

	// * Calculating phase using angle
	thrust::device_vector<comThr> phaseC(echo_num);
	phaseC[0] = comThr(1.0f, 0.0f);

	thrust::transform(thrust::device, angle.begin(), angle.end(), phaseC.begin() + 1, \
		[]__host__ __device__(const float& x) { return thrust::exp(comThr(0.0, -x)); });

	// * Compensation
	cuComplex* d_phaseC = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(phaseC.data()));

	dim3 block(256);  // block size
	dim3 grid((data_num + block.x - 1) / block.x);  // grid size

	Compensate_Phase << <grid, block >> > (d_data, d_phaseC, d_data, echo_num, range_num);
	checkCudaErrors(cudaDeviceSynchronize());
}


__global__ void Compensate_Phase(cuComplex* d_res, cuComplex* d_vec, cuComplex* d_data, int rows, int cols)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < rows * cols) {
		d_res[tid] = cuCmulf(d_vec[int(tid / cols)], d_data[tid]);
	}
}


void rangeVariantPhaseComp(cuComplex* d_data, const RadarParameters& paras, float* h_azimuth, float* h_pitch, cublasHandle_t handle)
{
	int data_num = paras.echo_num * paras.range_num;

	// 获取中心角度，并将角度信息传到设备上
	int mid_index = paras.echo_num / 2;
	float middle_azimuth = h_azimuth[mid_index];
	float middle_pitch = h_pitch[mid_index];

	float* d_azimuth = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_azimuth, sizeof(float) * paras.echo_num));
	checkCudaErrors(cudaMemcpy(d_azimuth, h_azimuth, sizeof(float) * paras.echo_num, cudaMemcpyHostToDevice));

	float* d_pitch = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_pitch, sizeof(float) * paras.echo_num));
	checkCudaErrors(cudaMemcpy(d_pitch, h_pitch, sizeof(float) * paras.echo_num, cudaMemcpyHostToDevice));

	// 参数设置
	float resolution = static_cast<float>(LIGHT_SPEED_h) / static_cast<float>(2 * paras.band_width);
	float wave_length = static_cast<float>(LIGHT_SPEED_h) / static_cast<float>(paras.fc);

	// 假设转动中心在一维像的中心，生成距离向量
	thrust::device_vector<float>range(paras.range_num);
	thrust::sequence(thrust::device, range.begin(), range.end(), -float(paras.range_num) / 2.0);

	thrust::transform(thrust::device, range.begin(), range.end(), range.begin(), \
		[=]__host__ __device__(const float& x) { return x * 2.0f * resolution / wave_length; });

	// 计算回波相对于中间回波的转角
	thrust::device_vector<float> theta(paras.echo_num);
	thrust::device_ptr<float> thr_azimuth = thrust::device_pointer_cast(d_azimuth);
	thrust::device_ptr<float> thr_pitch = thrust::device_pointer_cast(d_pitch);
	float mid_x = sinf(middle_pitch / 180 * PI_h);
	float mid_y = cosf(middle_pitch / 180 * PI_h) * cosf(middle_azimuth / 180 * PI_h);
	float mid_z = cosf(middle_pitch / 180 * PI_h) * sinf(middle_azimuth / 180 * PI_h);

	thrust::transform(thrust::device, thr_azimuth, thr_azimuth + paras.echo_num, thr_pitch, theta.begin(), \
		[=]__host__ __device__(const float& cur_azi, const float& cur_pit) 
	{
		float x = sinf(cur_pit / 180 * PI_h);
		float y = cosf(cur_pit / 180 * PI_h) * cosf(cur_azi / 180 * PI_h);
		float z = cosf(cur_pit / 180 * PI_h) * sinf(cur_azi / 180 * PI_h);

		float angle = (x * mid_x + y * mid_y + z * mid_z);
		angle = acosf(angle);
		float angle2;
		angle2 = powf(angle, 2.0);
		return angle2;
	});

	// 构建补偿矩阵并补偿距离像序列
	float* comp_mat = nullptr;
	checkCudaErrors(cudaMalloc((void**)&comp_mat, sizeof(float) * data_num));
	checkCudaErrors(cudaMemset(comp_mat, 0, sizeof(float) * data_num));
	
	// 08-05-2020 修改
	vecMulvec(handle, comp_mat, range, theta, 1.0f);
	thrust::device_ptr<float> thr_comp_mat = thrust::device_pointer_cast(comp_mat);
	thrust::device_ptr<comThr> thr_data = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_data));

	thrust::transform(thrust::device, thr_comp_mat, thr_comp_mat + data_num, thr_data, thr_data, \
		[]__host__ __device__(const float& x, const comThr& y) { return y * thrust::exp(comThr(0.0, -2 * PI_h * x)); });

	// Free Allocated GPU Memory
	checkCudaErrors(cudaFree(d_azimuth));
	checkCudaErrors(cudaFree(d_pitch));
	checkCudaErrors(cudaFree(comp_mat));
}


void fastEntropy(cuComplex* d_data, const int& echo_num, const int& range_num, cublasHandle_t handle)
{
	// 参数设置，新版数据格式是列主序，旧版是行主序，因此需要先转置
	// 列主序输入后续操作可能会简单一点，但懒得改了，春节后再改
	// 要改什么忘记了。。。
	// 08-06-2020改，d_data->d_data

	int data_num = echo_num * range_num;

	// * pre-processing and pre-imaging
	comThr* thr_DataTemp = reinterpret_cast<comThr*>(d_data);
	thrust::device_ptr<comThr> thr_Data = thrust::device_pointer_cast(thr_DataTemp);
	thrust::device_vector<float> thr_DataAbs(data_num);

	// abs(RetData_RA)
	thrust::transform(thrust::device, thr_Data, thr_Data + data_num, thr_DataAbs.begin(), \
		[]__host__ __device__(const comThr & a) { return thrust::abs(a); });

	thrust::device_vector<float> max_value(range_num);
	thrust::device_vector<int> max_index(range_num);
	// max_value = max(abs(RetData_RA),[],1);
	getMaxInColumns(thr_DataAbs, max_value, max_index, echo_num, range_num);

	// Test1<float>(max_value, range_num);
	// mean_value = mean(max_value);
	float init = 0.0f;
	float mean_value = thrust::transform_reduce(thrust::device, max_value.begin(), max_value.end(), \
		[=]__host__ __device__(const float& x) { return x / float(range_num); }\
		, init, thrust::plus<float>());

	float threshold = 1.48f * mean_value;
	// tgt_index = find(max_value>mean_value*1.48);
	thrust::device_vector<int>tgt_index(range_num);
	thrust::device_vector<int>::iterator end = thrust::copy_if(thrust::make_counting_iterator(0), thrust::make_counting_iterator(range_num), max_value.begin(), tgt_index.begin(), \
		[=]__host__ __device__(const float& x) { return (x > threshold); });
	
	int tgt_num = static_cast<int>(end - tgt_index.begin());
	tgt_index.resize(tgt_num);

	// tmpData=droptrace(RetData_RA);
	cuComplex* d_preComData{};
	checkCudaErrors(cudaMalloc((void**)&d_preComData, sizeof(cuComplex) * data_num));
	thrust::device_vector<comThr>PhaseC(echo_num);
	int isNeedCompensation = 1;
	dopplerTracking2(d_preComData, d_data, echo_num, range_num, PhaseC, isNeedCompensation);

	// image1=fft(tmpData);
	cuComplex* Image1{};
	checkCudaErrors(cudaMalloc((void**)&Image1, sizeof(cuComplex) * data_num));

	cufftHandle plan;

	int batch_img = range_num;
	int rank_img = 1;
	int n_img[1] = { echo_num };
	int inembed_img[] = { echo_num };
	int onembed_img[] = { echo_num };
	int istride_img = range_num;
	int ostride_img = range_num;
	int idist_img = 1;
	int odist_img = 1;

	checkCudaErrors(cufftPlanMany(&plan, rank_img, n_img,
		inembed_img, istride_img, idist_img,
		onembed_img, ostride_img, odist_img,
		CUFFT_C2C, batch_img));
	checkCudaErrors(cufftExecC2C(plan, d_preComData, Image1, CUFFT_FORWARD));

	int num_unit1 = tgt_num;
	int num_unit2 = nextPow2(tgt_num / 2) / 2;

	thrust::device_vector<float> sqrt_image(data_num);
	thrust::device_ptr<comThr> thr_image = thrust::device_pointer_cast(reinterpret_cast<comThr*>(Image1));

	// sqr_image = (abs(image2)).^2;
	thrust::transform(thrust::device, thr_image, thr_image + data_num, sqrt_image.begin(), \
		[]__host__ __device__(const comThr& x) { return powf(thrust::abs(x), 2); });

	// sum_image_vector = sum(sqr_image);
	float alpha = 1.0f;
	float beta = 0.0f;
	thrust::device_vector<float>Ones(echo_num, 1.0f);
	float* d_Ones = reinterpret_cast<float*>(thrust::raw_pointer_cast(Ones.data()));

	thrust::device_vector<float>sum_image_vector(range_num, 0.0f);
	float* d_sumImg = reinterpret_cast<float*>(thrust::raw_pointer_cast(sum_image_vector.data()));

	float* d_sqrtImg = reinterpret_cast<float*>(thrust::raw_pointer_cast(sqrt_image.data()));
	checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, range_num, echo_num, &alpha, d_sqrtImg, range_num, d_Ones, 1,
		&beta, d_sumImg, 1));

	// * find index of the first num_unit1 large values of sum_image_vector
	thrust::device_vector<float>sum_image_vector_copy(range_num, 0.0f);
	thrust::copy(thrust::device, sum_image_vector.begin(), sum_image_vector.end(), sum_image_vector_copy.begin());

	thrust::device_vector<int>value_sumImg(range_num);
	thrust::sequence(thrust::device, value_sumImg.begin(), value_sumImg.end(), 0);
	thrust::stable_sort_by_key(thrust::device, sum_image_vector_copy.begin(), sum_image_vector_copy.end(), value_sumImg.begin(), thrust::greater<float>());

	thrust::device_vector<int>select_bin1(num_unit1);
	thrust::copy(thrust::device, value_sumImg.begin(), value_sumImg.begin() + num_unit1, select_bin1.begin());

	// * get entropy
	thrust::device_vector<float>mask_entropy(data_num, 0.0f);
	float* d_mask_entropy = reinterpret_cast<float*>(thrust::raw_pointer_cast(mask_entropy.data()));

	checkCudaErrors(cublasSger(handle, range_num, echo_num, &alpha, d_sumImg, 1, d_Ones, 1, d_mask_entropy, range_num));

	thrust::transform(thrust::device, sqrt_image.begin(), sqrt_image.end(), mask_entropy.begin(), mask_entropy.begin(), thrust::divides<float>());


	thrust::transform(thrust::device, mask_entropy.begin(), mask_entropy.end(), mask_entropy.begin(), \
		[]__host__ __device__(const float& x) { return -(x * logf(x)); });

	thrust::device_vector<float>entropy(range_num, 0.0f);
	float* d_entropy = reinterpret_cast<float*>(thrust::raw_pointer_cast(entropy.data()));

	checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, range_num, echo_num, &alpha, d_mask_entropy, range_num, d_Ones, 1,
		&beta, d_entropy, 1));

	/************************
	* Maybe time consuming *
	************************/
	thrust::device_vector<float>selectEntropy;
	for (int ii = 0; ii < num_unit1; ii++) {
		selectEntropy.push_back(entropy[select_bin1[ii]]);
	}
	thrust::stable_sort_by_key(thrust::device, selectEntropy.begin(), selectEntropy.end(), select_bin1.begin());

	thrust::device_vector<int>select_bin2(num_unit2);
	thrust::copy(thrust::device, select_bin1.begin(), select_bin1.begin() + num_unit2, select_bin2.begin());
	thrust::sort(thrust::device, select_bin2.begin(), select_bin2.end());

	int* select_bin = reinterpret_cast<int*>(thrust::raw_pointer_cast(select_bin2.data()));

	// * rebuild echo
	/************************************
	* this kernel needs to be improved *
	************************************/
	cuComplex* newData = nullptr;
	checkCudaErrors(cudaMalloc((void**)&newData, sizeof(cuComplex) * num_unit2 * echo_num));
	unsigned int blockSize = num_unit2;
	unsigned int gridSize = echo_num;
	dim3 block(blockSize);
	dim3 grid(gridSize);

	Select_Rangebins <<<grid, block >>> (newData, d_data, select_bin, echo_num, range_num, num_unit2);

	// * doppler phase tracking
	isNeedCompensation = 0;
	thrust::device_vector<comThr>Phase_select(echo_num);
	dopplerTracking2(newData, newData, echo_num, num_unit2, Phase_select, isNeedCompensation);

	// * minimum entropy searching
	int search_num = 100;  // iteration numbers

	cuComplex* d_tempData = nullptr;    // tmpData (cuComplex and thrust)
	checkCudaErrors(cudaMalloc((void**)&d_tempData, echo_num * num_unit2 * sizeof(cuComplex)));
	comThr* thr_Temp = reinterpret_cast<comThr*>(d_tempData);
	thrust::device_ptr<comThr>thr_tempData = thrust::device_pointer_cast(thr_Temp);

	cuComplex* d_phi = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(Phase_select.data())); // Phase_select (thrust)

	comThr* thr_newData_temp = reinterpret_cast<comThr*>(newData);
	thrust::device_ptr<comThr>thr_newData = thrust::device_pointer_cast(thr_newData_temp);    // newData (thrust)

	thrust::device_vector<comThr>thr_rowSum(num_unit2, comThr(1.0f, 0.0f));    // all ones vector for sum(*,2)
	cuComplex* d_rowSum = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(thr_rowSum.data()));

	cuComplex alpha_c = make_cuComplex(1.0f, 0.0f);
	cuComplex beta_c = make_cuComplex(0.0f, 0.0f);

	int blockSize_com = 64;    // set block and grid for phase compensation
	int gridSize_com;
	//cudaOccupancyMaxPotentialBlockSize(&minGridSize_com, &blockSize_com, Compensate_Phase, 0, 0);
	gridSize_com = (num_unit2 * echo_num + blockSize_com - 1) / blockSize_com;
	dim3 grid_com(gridSize_com);
	dim3 block_com(blockSize_com);

	int batch = num_unit2;    // prepare for fft (along columns)
	int rank = 1;
	int n[1] = { echo_num };
	int inembed[] = { echo_num };
	int onembed[] = { echo_num };
	int istride = num_unit2;
	int ostride = num_unit2;
	int idist = 1;
	int odist = 1;

	checkCudaErrors(cufftPlanMany(&plan, rank, n,
		inembed, istride, idist,
		onembed, ostride, odist,
		CUFFT_C2C, batch));

	for (int ii = 0; ii < search_num; ii++) {

		Compensate_Phase <<<grid_com, block_com >>> (d_tempData, d_phi, newData, echo_num, num_unit2);

		checkCudaErrors(cufftExecC2C(plan, (cufftComplex*)d_tempData, (cufftComplex*)d_tempData, CUFFT_FORWARD));

		thrust::transform(thrust::device, thr_tempData, thr_tempData + echo_num * num_unit2, thr_tempData, \
			[]__host__ __device__(comThr& x) { return x * thrust::abs(x); });

		checkCudaErrors(cufftExecC2C(plan, (cufftComplex*)d_tempData, (cufftComplex*)d_tempData, CUFFT_INVERSE));

		thrust::transform(thrust::device, thr_newData, thr_newData + echo_num * num_unit2, thr_tempData, thr_tempData, \
			[=]__host__ __device__(const comThr& x, const comThr& y) { return (thrust::conj(x) * (y / float(echo_num))); });

		checkCudaErrors(cublasCgemv(handle, CUBLAS_OP_T, num_unit2, echo_num, &alpha_c, d_tempData, num_unit2, d_rowSum, 1, &beta_c, d_phi, 1));

		thrust::transform(thrust::device, Phase_select.begin(), Phase_select.end(), Phase_select.begin(), \
			[]__host__ __device__(const comThr& x) { return (x / thrust::abs(x)); });
	}

	gridSize_com = (data_num + blockSize_com - 1) / blockSize_com;
	dim3 grid_com2(gridSize_com);
	dim3 block_com2(blockSize_com);
	Compensate_Phase <<<grid_com2, block_com2 >>> (d_data, d_phi, d_data, echo_num, range_num);


	// * free space
	checkCudaErrors(cudaFree(Image1));
	checkCudaErrors(cudaFree(newData));
	checkCudaErrors(cudaFree(d_tempData));
	checkCudaErrors(cufftDestroy(plan));
}


void dopplerTracking2(cuComplex* d_preComData, cuComplex* d_data, const int& echo_num, const int& range_num, thrust::device_vector<comThr>& phaseC, int isNeedCompensation)
{
	// step 1: generate key array
	thrust::device_vector<int>oriKey(echo_num - 1);
	thrust::sequence(thrust::device, oriKey.begin(), oriKey.end(), 1);
	thrust::device_vector<int> d_counts(echo_num - 1, range_num);
	thrust::device_vector<int> keyVec((echo_num - 1) * range_num);    // 1,...,1,...,...,255,...,255

	// expand keys according to counts
	expand(d_counts.begin(), d_counts.end(),
		oriKey.begin(),
		keyVec.begin());


	// step 2: complex number multiple
	thrust::device_ptr<comThr> thr_d_data = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_data));

	thrust::device_vector<comThr> mulRes((echo_num - 1) * range_num);

	thrust::transform(thrust::device, thr_d_data, thr_d_data + (echo_num - 1) * range_num, thr_d_data + range_num, mulRes.begin(), \
		[]__host__ __device__(const comThr & x, const comThr & y) { return thrust::conj(x) * y; });

	// step 3: reduce 255 blocks
	thrust::device_vector<comThr> xw((echo_num - 1));

	thrust::reduce_by_key(thrust::device, keyVec.begin(), keyVec.end(), mulRes.begin(), thrust::make_discard_iterator(), xw.begin());

	// step 4: get angle
	thrust::device_vector<float> angle(echo_num - 1);
	thrust::transform(thrust::device, xw.begin(), xw.end(), angle.begin(), \
		[]__host__ __device__(const comThr& x) { return thrust::arg((x / thrust::abs(x))); });
	thrust::inclusive_scan(thrust::device, angle.begin(), angle.end(), angle.begin());

	// step 5: get compensation phase
	//thrust::device_vector<comThr> phaseC(echo_num); //!!!!!!!
	phaseC[0] = comThr(1.0f, 0.0f);

	thrust::transform(thrust::device, angle.begin(), angle.end(), phaseC.begin() + 1, \
		[]__host__ __device__(const float& x) { return thrust::exp(comThr(0.0, -x)); });

	// step 6: compensation phase
	if (isNeedCompensation) {
		cuComplex* d_phaseC = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(phaseC.data()));

		int blockSize = 64;
		//int minGridSize;
		int gridSize;
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Compensate_Phase, 0, 0);
		gridSize = (range_num * echo_num + blockSize - 1) / blockSize;
		dim3 grid(gridSize);
		dim3 block(blockSize);

		Compensate_Phase <<<grid, block >>> (d_preComData, d_phaseC, d_data, echo_num, range_num);
	}
}


__global__ void Select_Rangebins(cuComplex* newData, cuComplex* d_data, int* select_bin, int echo_num, int range_num, int num_unit2)
{
	unsigned int tid = threadIdx.x;
	unsigned int index = tid + blockIdx.x * blockDim.x;

	if (index >= echo_num * num_unit2)
		return;

	int rowNum = int(index / num_unit2);
	int reIndex = rowNum * range_num + select_bin[tid];

	newData[index] = d_data[reIndex];

}
