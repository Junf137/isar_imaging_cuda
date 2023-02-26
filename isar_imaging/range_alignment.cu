#include "range_alignment.cuh"

void RangeAlignment_linej(cuComplex* d_data, RadarParameters paras, thrust::device_vector<int>& fftshift_vec)
{
	// step 1: reverse HRRP to time domain signal
	int data_num = paras.num_echoes * paras.num_range_bins;

	thrust::device_ptr<comThr>thr_d_data = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_data));

	cufftHandle plan;
	checkCudaErrors(cufftPlan1d(&plan, paras.num_range_bins, CUFFT_C2C, paras.num_echoes));
	checkCudaErrors(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));    // IFFT没有除以N
	thrust::transform(thrust::device, thr_d_data, thr_d_data + data_num, fftshift_vec.begin(),
		thr_d_data, []__host__ __device__(thrust::complex<float> x, int y) { return x * float(y); });  // fftshift

	// step 2: 参数设置
	// * generating hamming window
	thrust::device_vector<float> hamming_window(paras.num_range_bins, 0.0f);
	thrust::sequence(thrust::device, hamming_window.begin(), hamming_window.begin() + paras.num_range_bins / 2, 0.0f);
	thrust::sequence(thrust::device, hamming_window.begin() + paras.num_range_bins / 2, hamming_window.end(), float(paras.num_range_bins / 2 - 1), -1.0f);
	thrust::transform(thrust::device, hamming_window.begin(), hamming_window.end(), hamming_window.begin(), \
		[=]__host__ __device__(const float& x) { return (0.54f - 0.46f * cos(2 * PI_h * (x / paras.num_range_bins - 1))); });


	// * 构造距离索引向量
	thrust::device_vector<float> vec_range(paras.num_range_bins);
	thrust::sequence(thrust::device, vec_range.begin(), vec_range.end(), 0.0f);

	// * 构造频谱中心化矩阵
	thrust::device_vector<comThr> shift_vector(paras.num_range_bins);
	thrust::transform(thrust::device, hamming_window.begin(), hamming_window.end(), vec_range.begin(), shift_vector.begin(), \
		[]__host__ __device__(const float& x, const float& c) { return (x * thrust::exp(comThr(0.0f, -PI_h * c))); });

	thrust::device_vector<comThr> ones_numEcho_1(paras.num_echoes, comThr(1.0f, 0.0f));
	// * 频谱中心化向量扩展为矩阵
	cuComplex* d_ones_shiftVecotr = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_ones_shiftVecotr, sizeof(cuComplex) * data_num));
	checkCudaErrors(cudaMemset(d_ones_shiftVecotr, 0, sizeof(float) * 2 * data_num));
	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));
	vectorMulvectorCublasC(handle, d_ones_shiftVecotr, shift_vector, ones_numEcho_1, paras.num_range_bins, paras.num_echoes);

	// * 处理频谱中心化
	comThr* thr_d_temp_ones_shiftVector = reinterpret_cast<comThr*>(d_ones_shiftVecotr);
	thrust::device_ptr<comThr> thr_ones_shiftVector = thrust::device_pointer_cast(thr_d_temp_ones_shiftVector);
	thrust::transform(thrust::device, thr_d_data, thr_d_data + data_num, thr_ones_shiftVector, thr_d_data, \
		[=]__host__ __device__(const comThr& a, const comThr& b) { return (a / static_cast<float>(paras.num_range_bins)) * b; });  // (a / N) * b, IFFT数据归一化后乘以另一复数

	// * 定义NN，即距离单元的一半
	int NN = paras.num_range_bins / 2;

	// step 3: 处理第一个回波
	// * 对第一个回波进行ifft（未归一化）
	checkCudaErrors(cufftPlan1d(&plan, paras.num_range_bins, CUFFT_C2C, 1));
	checkCudaErrors(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
	// * 确定模板向量
	thrust::transform(thrust::device, thr_d_data, thr_d_data + paras.num_range_bins, thr_d_data, \
		[=]__host__ __device__(const comThr& a) { return (a / static_cast<float>(paras.num_range_bins)); });
	
	thrust::device_vector<float> vec_a(paras.num_range_bins);
	thrust::transform(thrust::device, thr_d_data, thr_d_data + paras.num_range_bins, vec_a.begin(), \
		[]__host__ __device__(const comThr& a) { return thrust::abs(a); });


	thrust::device_vector<float> vec_b1(paras.num_range_bins);    // 模板向量b1
	thrust::copy(vec_a.begin(), vec_a.end(), vec_b1.begin());

	// step 4: 循环处理后续回波
	// * 初始化
	thrust::device_vector<comThr> ifft_temp(paras.num_range_bins, comThr(0.0f, 0.0f));
	cuComplex* d_ifft_temp = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(ifft_temp.data())); // 保存每次循环的ifft结果
	thrust::device_vector<float> vec_corr(paras.num_range_bins, 0.0f); // 保存每次循环的相关结果
	thrust::device_vector<comThr> a_out(paras.num_range_bins, comThr(0.0f, 0.0f)); // 每个回波对齐结果                      
	int maxPos;
	float xstar;
	float mopt;

	for (int ii = 1; ii < paras.num_echoes; ii++) {
		// * 计算abs(ifft(Data(n,:),N));
		checkCudaErrors(cufftExecC2C(plan, d_data + ii * paras.num_range_bins, d_ifft_temp, CUFFT_INVERSE));
		thrust::transform(thrust::device, ifft_temp.begin(), ifft_temp.end(), vec_a.begin(), \
			[=]__host__ __device__(const comThr& x) { return thrust::abs(x / static_cast<float>(paras.num_range_bins)); });  // ifft数据除以N后取绝对值

		// * 计算a和b1的相关函数
		GetCorrelation(vec_a, vec_b1, vec_corr);

		// * 求最大值索引
		thrust::device_vector<float>::iterator iter = \
			thrust::max_element(vec_corr.begin(), vec_corr.end());
		maxPos = static_cast<int>(iter - vec_corr.begin());

		// * 二项式拟合，得到最大值的精确位置
		xstar = BinomialFix(vec_corr, maxPos);
		mopt = maxPos + xstar - NN;

		// * 计算aout
		comThr* thr_d_dataiTemp = reinterpret_cast<comThr*>(d_data + ii * paras.num_range_bins);
		thrust::device_ptr<comThr> thr_data_i = thrust::device_pointer_cast(thr_d_dataiTemp);

		thrust::transform(thrust::device, vec_range.begin(), vec_range.end(), a_out.begin(), \
			[=]__host__ __device__(const float& x) { return thrust::exp(comThr(0.0f, -2 * PI_h * x * mopt / paras.num_range_bins)); });  // 构造频移向量

		thrust::transform(thrust::device, thr_data_i, thr_data_i + paras.num_range_bins, a_out.begin(), thr_data_i, \
			[]__host__ __device__(const comThr& a, const comThr& b) { return a * b; });

		checkCudaErrors(cufftExecC2C(plan, d_data + ii * paras.num_range_bins, d_data + ii * paras.num_range_bins, CUFFT_INVERSE));
		thrust::transform(thrust::device, thr_data_i, thr_data_i + paras.num_range_bins, thr_data_i, \
			[=]__host__ __device__(const comThr& a) { return (a / static_cast<float>(paras.num_range_bins)); });

		thrust::transform(thrust::device, vec_b1.begin(), vec_b1.end(), thr_data_i, vec_b1.begin(), \
			[]__host__ __device__(const float& x, const comThr& y) { return (x * 0.95f + thrust::abs(y)); });  // 用已经对齐的回波更新模板
	}

	// step 6: 释放空间
	checkCudaErrors(cudaFree(d_ones_shiftVecotr));
	checkCudaErrors(cublasDestroy(handle));
	checkCudaErrors(cufftDestroy(plan));
}


void GetCorrelation(thrust::device_vector<float>& vec_a, thrust::device_vector<float>& vec_b, thrust::device_vector<float>& vecCorr)
{
	int vecLen = static_cast<int>(vec_a.size());

	cufftHandle plan;

	// fft_b
	thrust::device_vector<comThr> fft_b(vecLen / 2 + 1, comThr(0.0f, 0.0f));
	cuComplex* d_fft_b = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(fft_b.data()));
	float* d_vec_b = reinterpret_cast<float*>(thrust::raw_pointer_cast(vec_b.data()));
	checkCudaErrors(cufftPlan1d(&plan, vecLen, CUFFT_R2C, 1));
	checkCudaErrors(cufftExecR2C(plan, d_vec_b, d_fft_b));  // implicitly forward

	// fft_a
	thrust::device_vector<comThr> fft_a(vecLen / 2 + 1, comThr(0.0f, 0.0f));
	cuComplex* d_fft_a = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(fft_a.data()));
	float* d_vec_a = reinterpret_cast<float*>(thrust::raw_pointer_cast(vec_a.data()));
	checkCudaErrors(cufftExecR2C(plan, d_vec_a, d_fft_a));

	// conj multiply, result store in fft_b
	thrust::transform(thrust::device, fft_a.begin(), fft_a.end(), fft_b.begin(), fft_b.begin(), \
		[]__host__ __device__(const comThr& x, const comThr& y) { return thrust::conj(x) * y; });

	// ifft
	float* d_vec_ifft = reinterpret_cast<float*>(thrust::raw_pointer_cast(vecCorr.data()));
	checkCudaErrors(cufftPlan1d(&plan, vecLen, CUFFT_C2R, 1));
	checkCudaErrors(cufftExecC2R(plan, d_fft_b, d_vec_ifft));  // implicitly inverse

	// fftshift
	thrust::swap_ranges(thrust::device, vecCorr.begin(), vecCorr.begin() + vecLen / 2, vecCorr.begin() + vecLen / 2);

	checkCudaErrors(cufftDestroy(plan));
}


float BinomialFix(thrust::device_vector<float>& vec_Corr, int maxPos)
{
	float f1 = vec_Corr[maxPos - 1];
	float f2 = vec_Corr[maxPos];
	float f3 = vec_Corr[maxPos + 1];

	float fa = (f1 + f3 - 2 * f2) / 2;
	float fb = (f3 - f1) / 2;
	//float fc = f2;

	float xstar = -fb / (2 * fa);
	return xstar;
}


void HRRPCenter(cuComplex* d_data, RadarParameters paras, const int inter_length)
{
	// * 求平均距离像（最大值归一)
	const int data_num = paras.num_echoes * paras.num_range_bins;
	// 类型转换：cuComplex->thrust
	thrust::device_ptr<comThr>thr_d_data = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_data));
	// 求包络，并归一化
	thrust::device_vector<float>HRRP0(data_num);
	thrust::transform(thrust::device, thr_d_data, thr_d_data + data_num, HRRP0.begin(), \
		[]__host__ __device__(const comThr & a) { return thrust::abs(a); });
	thrust::device_vector<float>::iterator iter = \
		thrust::max_element(HRRP0.begin(), HRRP0.end());
	float max_abs_hrrp = *iter;
	thrust::transform(thrust::device, HRRP0.begin(), HRRP0.end(), HRRP0.begin(), \
		[=]__host__ __device__(const float& x) { return float(x) / float(max_abs_hrrp); });  // 一般用于向量最大值归一化

	// 利用全1向量做乘法，求平均距离像
	thrust::device_vector<float>thr_ones_echo_num(paras.num_echoes, 1.0);
	float* ones_echo_num = reinterpret_cast<float*>(thrust::raw_pointer_cast(thr_ones_echo_num.data()));

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));
	float alpha = 1.0f / float(paras.num_echoes);
	float beta = 0.0;
	thrust::device_vector<float>ARP(paras.num_range_bins, 0.0);
	float* d_ARP = (float*)(thrust::raw_pointer_cast(ARP.data()));//*
	float* d_HRRP0 = (float*)(thrust::raw_pointer_cast(HRRP0.data()));

	//checkCudaErrors(cublasSgemv(handle,CUBLAS_OP_T,paras.num_echoes,paras.num_range_bins,&alpha,
	//	                          d_HRRP0,paras.num_echoes,ones_echo_num,1,&beta,d_ARP,1));
	checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, paras.num_range_bins, paras.num_echoes, &alpha,
		d_HRRP0, paras.num_range_bins, ones_echo_num, 1, &beta, d_ARP, 1));

	// * 寻找背景噪声阈值
	thrust::device_vector<float>ARP1(paras.num_range_bins);
	ARP1 = ARP;
	thrust::stable_sort(thrust::device, ARP1.begin(), ARP1.end());
	iter = thrust::max_element(ARP1.begin(), ARP1.end());
	float max_arp = *iter;
	const float extra_value = (float)inter_length * max_arp / (float)paras.num_range_bins;
	const unsigned int diff_length = paras.num_range_bins - inter_length;
	thrust::device_vector<float>diff(diff_length);
	thrust::transform(thrust::device, ARP1.begin() + inter_length, ARP1.end(), ARP1.begin(), diff.begin(), \
		[=]__host__ __device__(const float& x, const float& y) { return std::abs(((x - y - extra_value))); });  // 构造中心化函数中的diff向量

	iter = thrust::min_element(diff.begin(), diff.end());
	int min_diff_pos = static_cast<int>(iter - diff.begin());
	float low_threshold_gray = ARP1[min_diff_pos + static_cast<int>(inter_length / 2)];
	// 以下利用thrust实现 indices = find(ARP>low_threshold_gray);
	thrust::device_vector<int>indices(paras.num_range_bins);
	thrust::device_vector<int>::iterator end = thrust::copy_if(thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(int(paras.num_range_bins)),
		ARP.begin(),
		indices.begin(),
		thrust::placeholders::_1 > low_threshold_gray);
	int indices_length = static_cast<int>(end - indices.begin());
	indices.resize(indices_length);

	int WL = 8;    // MATLAB程序里的变量，意义未知
	float* d_ARP_ave;
	checkCudaErrors(cudaMalloc((void**)&d_ARP_ave, sizeof(float) * indices_length));
	//float *d_ARP = reinterpret_cast<float*>(thrust::raw_pointer_cast(ARP.data()));    // 类型转换：thrust->normal gpu array
	int* d_indices = (int*)(thrust::raw_pointer_cast(indices.data()));//*
	int block_size = 32;
	//int min_grid_size;
	int grid_size = (indices_length + block_size - 1) / block_size;
	dim3 grid(grid_size);
	dim3 block(block_size);
	GetARPMean <<< grid, block >>> (d_ARP_ave, d_indices, d_ARP, indices_length, WL, paras);

	thrust::device_ptr<float>ARP_ave(d_ARP_ave);

	// 以下实现ind=find(APR_ave<low_threshold_gray); 
	thrust::device_vector<int>ind(indices_length);
	thrust::device_vector<int>::iterator end_min = thrust::copy_if(thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(indices_length),
		ARP_ave,
		ind.begin(),
		thrust::placeholders::_1 < low_threshold_gray);
	int ind_length = static_cast<int>(end_min - ind.begin());
	ind.resize(ind_length);
	if (ind_length == indices_length)
		return;

	// 使用核函数实现indices(ind)=[];这里直接将索引为ind的置于0
	int* d_ind = (int*)(thrust::raw_pointer_cast(ind.data()));
	block_size = nextPow2(ind_length);    // 块大小扩为离ind_length最近的2的幂
	int set_num = 0;
	setNumInArray<<<1, block_size>>> (d_indices, d_ind, set_num, ind_length);

	int mean_indice = thrust::reduce(thrust::device, indices.begin(), indices.end(), 0, thrust::plus<int>()) / (indices_length - ind_length);

	int shift_num = mean_indice - paras.num_range_bins / 2 + 1;

	cufftHandle plan;
	int batch = paras.num_echoes;
	int rank = 1;
	int n[1] = { paras.num_range_bins };
	int inembed[] = { paras.num_range_bins };
	int onembed[] = { paras.num_range_bins };
	int istride = 1;
	int ostride = 1;
	int idist = paras.num_range_bins;
	int odist = paras.num_range_bins;
	checkCudaErrors(cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch));
	checkCudaErrors(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

	// 下面利用thrust构造exp(1j*2*pi*Vec*shiftnum) （长度为距离向点数）
	thrust::device_vector<comThr>shift_vector(paras.num_range_bins);
	thrust::device_vector<float>vec(paras.num_range_bins);
	thrust::sequence(thrust::device, vec.begin(), vec.end(), 0.0f);
	thrust::transform(thrust::device, vec.begin(), vec.end(), shift_vector.begin(), \
		[=]__host__ __device__(const float& x) 
	{ return (thrust::exp(comThr(0.0, 2 * PI_h * x * static_cast<float>(shift_num) / static_cast<float>(paras.num_range_bins)))); });  // 构造HRRPCenter最后的平移向量

	// 通过乘以全一向量实现repmat(exp(1j*2*pi*Vec*shiftnum),n,1)
	thrust::device_vector<comThr>shift_mat(paras.num_echoes * paras.num_range_bins, comThr(0.0f, 0.0f));
	cuComplex* d_shift_mat = (cuComplex*)(thrust::raw_pointer_cast(shift_mat.data()));//*
	thrust::device_vector<comThr>thr_com_ones_echo_num(paras.num_echoes, comThr(1.0f, 0.0f));
	//vectorMulvectorCublasC(handle, d_shift_mat, thr_com_ones_echo_num, shift_vector, paras.num_echoes, paras.num_range_bins);
	vectorMulvectorCublasC(handle, d_shift_mat, shift_vector, thr_com_ones_echo_num, paras.num_range_bins, paras.num_echoes);

	// 利用thrust实现点乘
	thrust::transform(thrust::device, shift_mat.begin(), shift_mat.end(), thr_d_data, thr_d_data, \
		[]__host__ __device__(const comThr& a, const comThr& b) { return a * b; });

	// ifft
	checkCudaErrors(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
	thrust::transform(thrust::device, thr_d_data, thr_d_data + data_num, thr_d_data, \
		[=]__host__ __device__(const comThr & a) { return (a / static_cast<float>(paras.num_range_bins)); });

	// * 释放空间
	checkCudaErrors(cublasDestroy(handle));
	checkCudaErrors(cudaFree(d_ARP_ave));
	checkCudaErrors(cufftDestroy(plan));
}


__global__ void GetARPMean(float* ARP_ave, int* indices, float* ARP, int indices_length, int WL, RadarParameters paras)
{
	unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	//for (int index_debug = tid; index_debug < indices_length; index_debug += blockDim.x*gridDim.x) {
	//	float temp_sum = 0;
	//	if (indices[index_debug] - WL / 2 >= 0 && indices[index_debug] + WL / 2 <= paras.num_range_bins - 1) {
	//		for (int index_ARP = indices[index_debug] - WL / 2; index_ARP <= indices[index_debug] + WL / 2; index_ARP++) {
	//			temp_sum += ARP[index_ARP];
	//		}
	//		temp_sum /= WL + 1;
	//	}
	//	else if (indices[index_debug] - WL / 2 < 0) {
	//		for (int index_ARP = 0; index_ARP <= indices[index_debug] + WL / 2; index_ARP++) {
	//			temp_sum += ARP[index_ARP];
	//		}
	//		temp_sum /= WL / 2 + indices[index_debug] + 1;
	//	}
	//	else if (indices[index_debug] + WL / 2 > paras.num_range_bins - 1) {
	//		for (int index_ARP = indices[index_debug] - WL / 2; index_ARP < paras.num_range_bins; index_ARP++) {
	//			temp_sum += ARP[index_ARP];
	//		}
	//		temp_sum /= (paras.num_range_bins) - (indices[index_debug] - WL / 2) - 1;
	//	}
	//	ARP_ave[index_debug] = temp_sum;
	//	printf("%d: %d\n",index_debug+1,indices[index_debug]);
	//}
	float temp_sum = 0;
	if (indices[tid] - WL / 2 >= 0 && indices[tid] + WL / 2 <= paras.num_range_bins - 1) {
		for (int index_ARP = indices[tid] - WL / 2; index_ARP <= indices[tid] + WL / 2; index_ARP++) {
			temp_sum += ARP[index_ARP];
		}
		temp_sum /= WL + 1;
	}
	else if (indices[tid] - WL / 2 < 0) {
		for (int index_ARP = 0; index_ARP <= indices[tid] + WL / 2; index_ARP++) {
			temp_sum += ARP[index_ARP];
		}
		temp_sum /= WL / 2 + indices[tid] + 1;
	}
	else if (indices[tid] + WL / 2 > paras.num_range_bins - 1) {
		for (int index_ARP = indices[tid] - WL / 2; index_ARP < paras.num_range_bins; index_ARP++) {
			temp_sum += ARP[index_ARP];
		}
		temp_sum /= (paras.num_range_bins) - (indices[tid] - WL / 2) - 1;
	}
	ARP_ave[tid] = temp_sum;

}

