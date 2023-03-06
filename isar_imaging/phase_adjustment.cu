#include "phase_adjustment.cuh"


/**************************************************
 * 函数功能：多普勒跟踪，实现相位粗校准
 * 输入参数:
 *     d_Data: 距离像序列(慢时间 * 快时间, 行主序)
 *     NumEcho: number of echos
 *     NumRange: number of fast-time points
 **************************************************/
void Doppler_Tracking(cuComplex* d_data, RadarParameters paras)
{
	int data_length = paras.echo_num * paras.range_num;
	//cuComplex *d_data_temp;
	//checkCudaErrors(cudaMalloc((void**)&d_data_temp, sizeof(cuComplex)*data_length));
	//checkCudaErrors(cudaMemset(d_data_temp, 0.0, sizeof(float) * 2 * data_length));

	//cublasHandle_t handle;    // 准备转置
	//checkCudaErrors(cublasCreate(&handle));
	//cuComplex alpha_trans_data;
	//alpha_trans_data.x = 1.0f;
	//alpha_trans_data.y = 0.0f;
	//cuComplex beta_trans_data;
	//beta_trans_data.x = 0.0f;
	//beta_trans_data.y = 0.0f;

	//checkCudaErrors(cublasCgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, paras.range_num, paras.echo_num, &alpha_trans_data,
	//	d_data, paras.echo_num, &beta_trans_data, d_data, paras.echo_num, d_data_temp, paras.range_num));    // 转为行主序

	// step 1: 为thrust::reduce_by_key生成键值
	thrust::device_vector<int>oriKey(paras.echo_num - 1);
	thrust::sequence(thrust::device, oriKey.begin(), oriKey.end(), 1);
	thrust::device_vector<int> d_counts(paras.echo_num - 1, paras.range_num);
	thrust::device_vector<int> keyVec((paras.echo_num - 1) * paras.range_num);    // 1,...,1,...,...,255,...,255

	// expand keys according to counts															  
	expand(d_counts.begin(), d_counts.end(),
		oriKey.begin(),
		keyVec.begin());


	// step 2: 复数相乘，即两两回波共轭相乘
	// type convert
	comThr* thr_d_temp_Data = reinterpret_cast<comThr*>(d_data);
	thrust::device_ptr<comThr> thr_d_Data = thrust::device_pointer_cast(thr_d_temp_Data);

	thrust::device_vector<comThr> mulRes((paras.echo_num - 1) * paras.range_num);

	thrust::transform(thrust::device, thr_d_Data, thr_d_Data + (paras.echo_num - 1) * paras.range_num, thr_d_Data + paras.range_num, mulRes.begin(), \
		[]__host__ __device__(const comThr & x, const comThr & y) { return thrust::conj(x) * y; });

	// step 3: 共轭相乘结果求和
	thrust::device_vector<comThr> xw((paras.echo_num - 1));
	thrust::reduce_by_key(thrust::device, keyVec.begin(), keyVec.end(), mulRes.begin(), thrust::make_discard_iterator(), xw.begin());


	// step 4: 求需要补偿的相角
	thrust::device_vector<float> angle(paras.echo_num - 1);
	Get_Angle op_ga;
	thrust::transform(thrust::device, xw.begin(), xw.end(), angle.begin(), op_ga);
	thrust::inclusive_scan(thrust::device, angle.begin(), angle.end(), angle.begin());

	// step 5: 由角度得到相位
	thrust::device_vector<comThr> phaseC(paras.echo_num);
	phaseC[0] = comThr(1.0f, 0.0f);
	Get_Com_Phase op_gcp;
	thrust::transform(thrust::device, angle.begin(), angle.end(), phaseC.begin() + 1, op_gcp);

	// step 6: 补偿
	cuComplex* d_phaseC = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(phaseC.data()));

	int blockSize = 128;
	//int minGridSize;
	int gridSize;
	//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Compensate_Phase, 0, 0);
	gridSize = (data_length + blockSize - 1) / blockSize;
	dim3 grid(gridSize);
	dim3 block(blockSize);

	Compensate_Phase <<<grid, block >>> (d_data, d_phaseC, d_data, paras.echo_num, paras.range_num);
	checkCudaErrors(cudaDeviceSynchronize());
	//checkCudaErrors(cublasCgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, paras.echo_num, paras.range_num, &alpha_trans_data,
	//	d_data_temp, paras.range_num, &beta_trans_data, d_data_temp, paras.range_num, d_data, paras.echo_num));    // 转回列主序

	//checkCudaErrors(cudaFree(d_data_temp));
	//checkCudaErrors(cublasDestroy(handle));
}


/******************************************
 * 函数功能: d_res = d_data * diag(d_vec)
 * 输入参数:
 ******************************************/
__global__ void Compensate_Phase(cuComplex* d_res, cuComplex* d_vec, cuComplex* d_data, int rows, int cols)
{
	unsigned int tid = threadIdx.x;
	unsigned int index = tid + blockIdx.x * blockDim.x;

	if (index >= rows * cols)
		return;

	d_res[index] = cuCmulf(d_vec[int(index / cols)], d_data[index]);
}


/****************************************************************
 * 函数功能:补偿大带宽下，因大转角引起的随距离空变的二次相位误差
 * 输入参数：
 *     d_data:       距离像序列（方位*距离，行主序）；
 *     paras：       雷达系统参数；
 *     azimuth_data: 窄带信息，方位角度,CPU上；
 *     pitch_data：  窄带信息，俯仰角度,CPU上。
 ****************************************************************/
void RangeVariantPhaseComp(cuComplex* d_data, RadarParameters paras, float* azimuth_data, float* pitch_data)
{
	// 获取中心角度，并将角度信息传到设备上
	int mid_index = paras.echo_num / 2;
	float middle_azimuth = azimuth_data[mid_index];
	float middle_pitch = pitch_data[mid_index];

	float* d_azimuth;
	checkCudaErrors(cudaMalloc((void**)&d_azimuth, sizeof(float) * paras.echo_num));
	checkCudaErrors(cudaMemcpy(d_azimuth, azimuth_data, sizeof(float) * paras.echo_num, cudaMemcpyHostToDevice));

	float* d_pitch;
	checkCudaErrors(cudaMalloc((void**)&d_pitch, sizeof(float) * paras.echo_num));
	checkCudaErrors(cudaMemcpy(d_pitch, pitch_data, sizeof(float) * paras.echo_num, cudaMemcpyHostToDevice));

	// 参数设置
	const int light_speed = 300000000;
	float resolution = float(light_speed) / float(2 * paras.band_width);
	float wave_length = float(light_speed) / float(paras.fc);

	// 假设转动中心在一维像的中心，生成距离向量
	thrust::device_vector<float>range(paras.range_num);
	thrust::sequence(thrust::device, range.begin(), range.end(), -float(paras.range_num) / 2.0);
	thrust::transform(thrust::device, range.begin(), range.end(), range.begin(), BuildRangeVector(resolution, wave_length));

	// 计算回波相对于中间回波的转角
	//int middle_echo_index = int(paras.echo_num / 2);
	thrust::device_vector<float>theta(paras.echo_num);
	thrust::device_ptr<float>thr_azimuth(d_azimuth);
	thrust::device_ptr<float>thr_pitch(d_pitch);
	float mid_x = sinf(middle_pitch / 180 * PI_h);
	float mid_y = cosf(middle_pitch / 180 * PI_h) * cosf(middle_azimuth / 180 * PI_h);
	float mid_z = cosf(middle_pitch / 180 * PI_h) * sinf(middle_azimuth / 180 * PI_h);
	thrust::transform(thrust::device, thr_azimuth, thr_azimuth + paras.echo_num, thr_pitch, theta.begin(), GetAngle(mid_x, mid_y, mid_z));

	// 构建补偿矩阵并补偿距离像序列
	float* comp_mat = nullptr;
	checkCudaErrors(cudaMalloc((void**)&comp_mat, sizeof(float) * paras.echo_num * paras.range_num));
	checkCudaErrors(cudaMemset(comp_mat, 0, sizeof(float) * paras.echo_num * paras.range_num));
	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));
	//vectorMulvectorCublasf(handle, comp_mat, theta, range, paras.echo_num, paras.range_num);
	// 08-05-2020 修改
	vecMulvec(handle, comp_mat, range, theta, 1.0f);
	thrust::device_ptr<float>thr_comp_mat(comp_mat);
	//thrust::device_vector<comThr>comp_phase(paras.echo_num*paras.range_num);
	comThr* thr_data_temp = reinterpret_cast<comThr*>(d_data);
	thrust::device_ptr<comThr>thr_data = thrust::device_pointer_cast(thr_data_temp);

	thrust::transform(thrust::device, thr_comp_mat, thr_comp_mat + paras.echo_num * paras.range_num, thr_data, thr_data, CompensatePhase());

	checkCudaErrors(cudaFree(d_azimuth));
	checkCudaErrors(cudaFree(d_pitch));
	checkCudaErrors(cudaFree(comp_mat));
	checkCudaErrors(cublasDestroy(handle));
}


/********************************************************************************
 * 函数功能:最小熵快速自聚焦
 *          参考：邱晓辉《ISAR成像快速最小熵相位补偿方法》，电子与信息学报，2004
 * 输入参数：
 *    d_data: 距离像序列(方位*距离，行主序）；
 *    paras:  雷达参数结构体
 ********************************************************************************/
void Fast_Entropy(cuComplex* d_data, RadarParameters paras)
{
	// 参数设置，新版数据格式是列主序，旧版是行主序，因此需要先转置
	// 列主序输入后续操作可能会简单一点，但懒得改了，春节后再改
	// 要改什么忘记了。。。
	// 08-06-2020改，d_Data->d_data
	int NumRange = paras.range_num;
	int NumEcho = paras.echo_num;
	//int data_length = paras.echo_num * paras.range_num;

	//cuComplex *d_Data;
	//checkCudaErrors(cudaMalloc((void**)&d_Data, sizeof(cuComplex)*data_length));
	//checkCudaErrors(cudaMemset(d_Data, 0.0, sizeof(float) * 2 * data_length));

	cublasHandle_t handle;    // 准备转置
	checkCudaErrors(cublasCreate(&handle));
	//cuComplex alpha_trans_data;
	//alpha_trans_data.x = 1.0f;
	//alpha_trans_data.y = 0.0f;
	//cuComplex beta_trans_data;
	//beta_trans_data.x = 0.0f;
	//beta_trans_data.y = 0.0f;

	//checkCudaErrors(cublasCgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, paras.range_num, paras.echo_num, &alpha_trans_data,
	//	d_data, paras.echo_num, &beta_trans_data, d_data, paras.echo_num, d_Data, paras.range_num));    // 转为行主序

	// * pre-processing and pre-imaging
	comThr* thr_DataTemp = reinterpret_cast<comThr*>(d_data);
	thrust::device_ptr<comThr> thr_Data = thrust::device_pointer_cast(thr_DataTemp);
	thrust::device_vector<float> thr_DataAbs(NumRange * NumEcho);

	// abs(RetData_RA)
	thrust::transform(thrust::device, thr_Data, thr_Data + NumEcho * NumRange, thr_DataAbs.begin(), \
		[]__host__ __device__(const comThr & a) { return thrust::abs(a); });

	thrust::device_vector<float> max_value(NumRange);
	thrust::device_vector<int> max_index(NumRange);
	// max_value = max(abs(RetData_RA),[],1);
	getMaxInColumns(thr_DataAbs, max_value, max_index, NumEcho, NumRange);

	// Test1<float>(max_value, NumRange);
	// mean_value = mean(max_value);
	float init = 0.0f;
	float mean_value = thrust::transform_reduce(thrust::device, max_value.begin(), max_value.end(), Normlize(NumRange), init, thrust::plus<float>());
	float threshold = 1.48f * mean_value;
	// tgt_index = find(max_value>mean_value*1.48);
	thrust::device_vector<int>tgt_index(NumRange);
	thrust::device_vector<int>::iterator end = \
		thrust::copy_if(thrust::make_counting_iterator(0),
			thrust::make_counting_iterator(NumRange),
			max_value.begin(),
			tgt_index.begin(),
			isWithinThreshold(threshold));
	int tgt_num = static_cast<int>(end - tgt_index.begin());
	tgt_index.resize(tgt_num);

	// tmpData=droptrace(RetData_RA);
	cuComplex* d_preComData;
	checkCudaErrors(cudaMalloc((void**)&d_preComData, sizeof(cuComplex) * NumEcho * NumRange));
	thrust::device_vector<comThr>PhaseC(NumEcho);
	int isNeedCompensation = 1;
	Doppler_Tracking2(d_preComData, d_data, NumEcho, NumRange, PhaseC, isNeedCompensation);

	// image1=fft(tmpData);
	cuComplex* Image1;
	checkCudaErrors(cudaMalloc((void**)&Image1, sizeof(cuComplex) * NumEcho * NumRange));

	cufftHandle plan;

	int batch_img = NumRange;
	int rank_img = 1;
	int n_img[1] = { NumEcho };
	int inembed_img[] = { NumEcho };
	int onembed_img[] = { NumEcho };
	int istride_img = NumRange;
	int ostride_img = NumRange;
	int idist_img = 1;
	int odist_img = 1;

	checkCudaErrors(cufftPlanMany(&plan, rank_img, n_img,
		inembed_img, istride_img, idist_img,
		onembed_img, ostride_img, odist_img,
		CUFFT_C2C, batch_img));
	checkCudaErrors(cufftExecC2C(plan, d_preComData, Image1, CUFFT_FORWARD));

	int num_unit1 = tgt_num;
	int num_unit2 = nextPow2(tgt_num / 2) / 2;

	thrust::device_vector<float>sqrt_image(NumRange * NumEcho, 0.0f);
	comThr* Image_temp = reinterpret_cast<comThr*>(Image1);
	thrust::device_ptr<comThr>thr_image = thrust::device_pointer_cast(Image_temp);
	// sqr_image = (abs(image2)).^2;
	absSquare op_as;
	thrust::transform(thrust::device, thr_image, thr_image + NumEcho * NumRange, sqrt_image.begin(), op_as);

	// sum_image_vector = sum(sqr_image);
	float alpha = 1.0f;
	float beta = 0.0f;
	thrust::device_vector<float>Ones(NumEcho, 1.0f);
	float* d_Ones = reinterpret_cast<float*>(thrust::raw_pointer_cast(Ones.data()));

	thrust::device_vector<float>sum_image_vector(NumRange, 0.0f);
	float* d_sumImg = reinterpret_cast<float*>(thrust::raw_pointer_cast(sum_image_vector.data()));

	float* d_sqrtImg = reinterpret_cast<float*>(thrust::raw_pointer_cast(sqrt_image.data()));
	checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, NumRange, NumEcho, &alpha, d_sqrtImg, NumRange, d_Ones, 1,
		&beta, d_sumImg, 1));

	// * find index of the first num_unit1 large values of sum_image_vector
	thrust::device_vector<float>sum_image_vector_copy(NumRange, 0.0f);
	thrust::copy(thrust::device, sum_image_vector.begin(), sum_image_vector.end(), sum_image_vector_copy.begin());

	thrust::device_vector<int>value_sumImg(NumRange);
	thrust::sequence(thrust::device, value_sumImg.begin(), value_sumImg.end(), 0);
	thrust::stable_sort_by_key(thrust::device, sum_image_vector_copy.begin(), sum_image_vector_copy.end(), value_sumImg.begin(), thrust::greater<float>());

	thrust::device_vector<int>select_bin1(num_unit1);
	thrust::copy(thrust::device, value_sumImg.begin(), value_sumImg.begin() + num_unit1, select_bin1.begin());

	// * get entropy
	thrust::device_vector<float>mask_entropy(NumRange * NumEcho, 0.0f);
	float* d_mask_entropy = reinterpret_cast<float*>(thrust::raw_pointer_cast(mask_entropy.data()));

	checkCudaErrors(cublasSger(handle, NumRange, NumEcho, &alpha, d_sumImg, 1, d_Ones, 1, d_mask_entropy, NumRange));

	thrust::transform(thrust::device, sqrt_image.begin(), sqrt_image.end(), mask_entropy.begin(), mask_entropy.begin(), thrust::divides<float>());

	GetEntropy op_ge;
	thrust::transform(thrust::device, mask_entropy.begin(), mask_entropy.end(), mask_entropy.begin(), op_ge);

	thrust::device_vector<float>entropy(NumRange, 0.0f);
	float* d_entropy = reinterpret_cast<float*>(thrust::raw_pointer_cast(entropy.data()));

	checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, NumRange, NumEcho, &alpha, d_mask_entropy, NumRange, d_Ones, 1,
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
	checkCudaErrors(cudaMalloc((void**)&newData, sizeof(cuComplex) * num_unit2 * NumEcho));
	unsigned int blockSize = num_unit2;
	unsigned int gridSize = NumEcho;
	dim3 block(blockSize);
	dim3 grid(gridSize);

	Select_Rangebins <<<grid, block >>> (newData, d_data, select_bin, NumEcho, NumRange, num_unit2);

	// * doppler phase tracking
	isNeedCompensation = 0;
	thrust::device_vector<comThr>Phase_select(NumEcho);
	Doppler_Tracking2(newData, newData, NumEcho, num_unit2, Phase_select, isNeedCompensation);

	// * minimum entropy searching
	int search_num = 100;  // iteration numbers

	cuComplex* d_tempData = nullptr;    // tmpData (cuComplex and thrust)
	checkCudaErrors(cudaMalloc((void**)&d_tempData, NumEcho * num_unit2 * sizeof(cuComplex)));
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
	//int minGridSize_com;
	int gridSize_com;
	//cudaOccupancyMaxPotentialBlockSize(&minGridSize_com, &blockSize_com, Compensate_Phase, 0, 0);
	gridSize_com = (num_unit2 * NumEcho + blockSize_com - 1) / blockSize_com;
	dim3 grid_com(gridSize_com);
	dim3 block_com(blockSize_com);

	int batch = num_unit2;    // prepare for fft (along columns)
	int rank = 1;
	int n[1] = { NumEcho };
	int inembed[] = { NumEcho };
	int onembed[] = { NumEcho };
	int istride = num_unit2;
	int ostride = num_unit2;
	int idist = 1;
	int odist = 1;

	checkCudaErrors(cufftPlanMany(&plan, rank, n,
		inembed, istride, idist,
		onembed, ostride, odist,
		CUFFT_C2C, batch));
	Complex_mul_absComplex op_cma;
	Elementwise_normalize op_en;
	for (int ii = 0; ii < search_num; ii++) {

		Compensate_Phase <<<grid_com, block_com >>> (d_tempData, d_phi, newData, NumEcho, num_unit2);

		checkCudaErrors(cufftExecC2C(plan, (cufftComplex*)d_tempData, (cufftComplex*)d_tempData, CUFFT_FORWARD));

		thrust::transform(thrust::device, thr_tempData, thr_tempData + NumEcho * num_unit2, thr_tempData, op_cma);

		checkCudaErrors(cufftExecC2C(plan, (cufftComplex*)d_tempData, (cufftComplex*)d_tempData, CUFFT_INVERSE));

		thrust::transform(thrust::device, thr_newData, thr_newData + NumEcho * num_unit2, thr_tempData, thr_tempData, Conj_mul_ifftNor(NumEcho));

		checkCudaErrors(cublasCgemv(handle, CUBLAS_OP_T, num_unit2, NumEcho, &alpha_c, d_tempData, num_unit2, d_rowSum, 1, &beta_c, d_phi, 1));

		thrust::transform(thrust::device, Phase_select.begin(), Phase_select.end(), Phase_select.begin(), op_en);

	}

	gridSize_com = (NumRange * NumEcho + blockSize_com - 1) / blockSize_com;
	dim3 grid_com2(gridSize_com);
	dim3 block_com2(blockSize_com);
	Compensate_Phase <<<grid_com2, block_com2 >>> (d_data, d_phi, d_data, NumEcho, NumRange);


	//checkCudaErrors(cublasCgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, paras.echo_num, paras.range_num, &alpha_trans_data,
	//	d_Data, paras.range_num, &beta_trans_data, d_Data, paras.range_num, d_data, paras.echo_num));    // 转回列主序

	// * free space
	checkCudaErrors(cudaFree(Image1));
	checkCudaErrors(cudaFree(newData));
	checkCudaErrors(cudaFree(d_tempData));
	//checkCudaErrors(cudaFree(d_Data));
	checkCudaErrors(cublasDestroy(handle));
	checkCudaErrors(cufftDestroy(plan));
}

// ---------------- Doppler centroid tracking ---------------- //
/*
* Function: realize Doppler centroid tracking to autofocus
* Paras:
* d_Data: Range profile(slow-time * fast-time, row major)
* NumEcho: number of echos
* NumRange: number of fast-time points
*/
void Doppler_Tracking2(cuComplex* d_preComData, cuComplex* d_Data, int NumEcho, int NumRange, thrust::device_vector<comThr>& phaseC, int isNeedCompensation)
{
	// step 1: generate key array
	thrust::device_vector<int>oriKey(NumEcho - 1);
	thrust::sequence(thrust::device, oriKey.begin(), oriKey.end(), 1);
	thrust::device_vector<int> d_counts(NumEcho - 1, NumRange);
	thrust::device_vector<int> keyVec((NumEcho - 1) * NumRange);    // 1,...,1,...,...,255,...,255

	// expand keys according to counts
	expand(d_counts.begin(), d_counts.end(),
		oriKey.begin(),
		keyVec.begin());


	// step 2: complex number multiple
	// type convert
	comThr* thr_d_temp_Data = reinterpret_cast<comThr*>(d_Data);
	thrust::device_ptr<comThr> thr_d_Data = thrust::device_pointer_cast(thr_d_temp_Data);

	thrust::device_vector<comThr> mulRes((NumEcho - 1) * NumRange);

	thrust::transform(thrust::device, thr_d_Data, thr_d_Data + (NumEcho - 1) * NumRange, thr_d_Data + NumRange, mulRes.begin(), \
		[]__host__ __device__(const comThr & x, const comThr & y) { return thrust::conj(x) * y; });

	// step 3: reduce 255 blocks
	thrust::device_vector<comThr> xw((NumEcho - 1));

	thrust::reduce_by_key(thrust::device, keyVec.begin(), keyVec.end(), mulRes.begin(), thrust::make_discard_iterator(), xw.begin());

	// step 4: get angle
	thrust::device_vector<float> angle(NumEcho - 1);
	Get_Angle op_ga;
	thrust::transform(thrust::device, xw.begin(), xw.end(), angle.begin(), op_ga);
	thrust::inclusive_scan(thrust::device, angle.begin(), angle.end(), angle.begin());

	// step 5: get compensation phase
	//thrust::device_vector<comThr> phaseC(NumEcho); //!!!!!!!
	phaseC[0] = comThr(1.0f, 0.0f);
	Get_Com_Phase op_gcp;
	thrust::transform(thrust::device, angle.begin(), angle.end(), phaseC.begin() + 1, op_gcp);

	// step 6: compensation phase
	if (isNeedCompensation) {
		cuComplex* d_phaseC = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(phaseC.data()));

		int blockSize = 64;
		//int minGridSize;
		int gridSize;
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Compensate_Phase, 0, 0);
		gridSize = (NumRange * NumEcho + blockSize - 1) / blockSize;
		dim3 grid(gridSize);
		dim3 block(blockSize);

		Compensate_Phase <<<grid, block >>> (d_preComData, d_phaseC, d_Data, NumEcho, NumRange);
	}
}


/********************************************************************************
* 函数功能:最小熵快速自聚焦中提取距离单元重建回波
*          参考：邱晓辉《ISAR成像快速最小熵相位补偿方法》，电子与信息学报，2004
* 输入参数：
*    newData:    提取距离单元后的回波
*    d_data:     距离像序列(方位*距离，列主序）、
*    select_bin: 选出距离单元的索引序列
*    NumEcho:    回波数
*    NumRange:   距离向采样点数
*    num_unit2： 选出的距离单元个数
********************************************************************************/
__global__ void Select_Rangebins(cuComplex* newData, cuComplex* d_Data, int* select_bin, int NumEcho, int NumRange, int num_unit2)
{
	unsigned int tid = threadIdx.x;
	unsigned int index = tid + blockIdx.x * blockDim.x;

	if (index >= NumEcho * num_unit2)
		return;

	int rowNum = int(index / num_unit2);
	int reIndex = rowNum * NumRange + select_bin[tid];

	newData[index] = d_Data[reIndex];

}
