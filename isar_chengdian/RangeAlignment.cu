#include "RangeAlignment.cuh"
//#include "cuda_occupancy.h"


/*******************************************
 * 函数功能:采用linej算法的距离对准函数
 * 参考程序:juliduizhun_modified2.m
 * 输入参数:
 * data: 距离像序列，存放在设备上
	   （数据格式：慢时间*快时间，行主序存放）
 * Paras：雷达参数结构体
 * shift_vec:fftshift向量
 *******************************************/
void RangeAlignment_linej(cuComplex* data, RadarParameters Paras, thrust::device_vector<int>& shift_vec)
{
	// step 1: 将距离像序列转为时域回波
	int data_length = Paras.num_echoes * Paras.num_range_bins;

	comThr* thr_d_data_temp = reinterpret_cast<comThr*>(data);
	thrust::device_ptr<comThr>thr_data = thrust::device_pointer_cast(thr_d_data_temp);

	cufftHandle plan;
	int batch = Paras.num_echoes;
	int rank = 1;
	int n[1] = { Paras.num_range_bins };
	int inembed[] = { Paras.num_range_bins };
	int onembed[] = { Paras.num_range_bins };
	int istride = 1;
	int ostride = 1;
	int idist = Paras.num_range_bins;
	int odist = Paras.num_range_bins;
	checkCuFFTErrors(cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch));
	checkCuFFTErrors(cufftExecC2C(plan, data, data, CUFFT_INVERSE));    // IFFT没有除以N
	thrust::transform(thrust::device, thr_data, thr_data + Paras.num_echoes * Paras.num_range_bins, shift_vec.begin(),
		thr_data, []__host__ __device__(thrust::complex<float> x, int y) { return x * float(y); });  // fftshift

	// step 2: 参数设置
	// * 生成hamming窗
	thrust::device_vector<float>hamming_window(Paras.num_range_bins, 0.0f);
	thrust::sequence(thrust::device, hamming_window.begin(), hamming_window.begin() + Paras.num_range_bins / 2, 0.0f);
	thrust::sequence(thrust::device, hamming_window.begin() + Paras.num_range_bins / 2, hamming_window.end(), float(Paras.num_range_bins / 2 - 1), -1.0f);
	thrust::transform(thrust::device, hamming_window.begin(), hamming_window.end(), hamming_window.begin(), Hamming_Window(Paras.num_range_bins));

	// * 构造距离索引向量
	thrust::device_vector<float>vec_range(Paras.num_range_bins);
	thrust::sequence(thrust::device, vec_range.begin(), vec_range.end(), 0.0f);

	// * 构造频谱中心化矩阵
	thrust::device_vector<comThr>shift_vector(Paras.num_range_bins);
	thrust::transform(thrust::device, hamming_window.begin(), hamming_window.end(), vec_range.begin(), shift_vector.begin(), GenerateShiftVector());
	thrust::device_vector<comThr> ones_numEcho_1(Paras.num_echoes, comThr(1.0f, 0.0f));
	// * 频谱中心化向量扩展为矩阵
	cuComplex* d_ones_shiftVecotr;
	checkCudaErrors(cudaMalloc((void**)&d_ones_shiftVecotr, sizeof(cuComplex) * data_length));
	checkCudaErrors(cudaMemset(d_ones_shiftVecotr, 0.0f, sizeof(float) * 2 * data_length));
	cublasHandle_t handle;
	checkCublasErrors(cublasCreate(&handle));
	vectorMulvectorCublasC(handle, d_ones_shiftVecotr, shift_vector, ones_numEcho_1, Paras.num_range_bins, Paras.num_echoes);

	// * 处理频谱中心化
	comThr* thr_d_temp_ones_shiftVector = reinterpret_cast<comThr*>(d_ones_shiftVecotr);
	thrust::device_ptr<comThr> thr_ones_shiftVector = thrust::device_pointer_cast(thr_d_temp_ones_shiftVector);
	thrust::transform(thrust::device, thr_data, thr_data + data_length, thr_ones_shiftVector, thr_data, ComplexIfft_Mul_Complex(Paras.num_range_bins));

	// * 定义NN，即距离单元的一半
	int NN = Paras.num_range_bins / 2;

	// step 3: 处理第一个回波
	// * 对第一个回波进行ifft（未归一化）
	checkCuFFTErrors(cufftPlan1d(&plan, Paras.num_range_bins, CUFFT_C2C, 1));
	checkCuFFTErrors(cufftExecC2C(plan, data, data, CUFFT_INVERSE));
	// * 确定模板向量
	thrust::transform(thrust::device, thr_data, thr_data + Paras.num_range_bins, thr_data, ifftNor(Paras.num_range_bins));
	thrust::device_vector<float>vec_a(Paras.num_range_bins);
	thrust::transform(thrust::device, thr_data, thr_data + Paras.num_range_bins, vec_a.begin(), Get_abs());
	thrust::device_vector<float>vec_b1(Paras.num_range_bins);    // 模板向量b1
	thrust::copy(vec_a.begin(), vec_a.end(), vec_b1.begin());

	// step 4: 循环处理后续回波
	// * 初始化
	thrust::device_vector<comThr>ifft_temp(Paras.num_range_bins, comThr(0.0f, 0.0f));
	cuComplex* d_ifft_temp = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(ifft_temp.data()));    // 保存每次循环的ifft结果
	thrust::device_vector<float>vec_corr(Paras.num_range_bins, 0.0f);                                     // 保存每次循环的相关结果
	thrust::device_vector<comThr>a_out(Paras.num_range_bins, comThr(0.0f, 0.0f));                         // 每个回波对齐结果                      
	unsigned int maxPos;
	float xstar;
	float mopt;

	for (int ii = 1; ii < Paras.num_echoes; ii++) {
		// * 计算abs(ifft(Data(n,:),N));
		checkCuFFTErrors(cufftExecC2C(plan, data + ii * Paras.num_range_bins, d_ifft_temp, CUFFT_INVERSE));
		thrust::transform(thrust::device, ifft_temp.begin(), ifft_temp.end(), vec_a.begin(), ifftNor_abs(Paras.num_range_bins));

		// * 计算a和b1的相关函数
		GetCorrelation(vec_a, vec_b1, vec_corr, Paras.num_range_bins);

		// * 求最大值索引
		thrust::device_vector<float>::iterator iter = \
			thrust::max_element(vec_corr.begin(), vec_corr.end());
		maxPos = iter - vec_corr.begin();

		// * 二项式拟合，得到最大值的精确位置
		xstar = BinomialFix(vec_corr, maxPos);
		mopt = maxPos + xstar - NN;

		// * 计算aout
		comThr* thr_d_dataiTemp = reinterpret_cast<comThr*>(data + ii * Paras.num_range_bins);
		thrust::device_ptr<comThr> thr_data_i = thrust::device_pointer_cast(thr_d_dataiTemp);

		thrust::transform(thrust::device, vec_range.begin(), vec_range.end(), a_out.begin(), Fre_shift(Paras.num_range_bins, mopt));
		thrust::transform(thrust::device, thr_data_i, thr_data_i + Paras.num_range_bins, a_out.begin(), thr_data_i, Complex_Mul_Complex());

		checkCuFFTErrors(cufftExecC2C(plan, data + ii * Paras.num_range_bins, data + ii * Paras.num_range_bins, CUFFT_INVERSE));
		thrust::transform(thrust::device, thr_data_i, thr_data_i + Paras.num_range_bins, thr_data_i, ifftNor(Paras.num_range_bins));

		thrust::transform(thrust::device, vec_b1.begin(), vec_b1.end(), thr_data_i, vec_b1.begin(), Renew_b1());
	}

	// step 6: 释放空间
	checkCudaErrors(cudaFree(d_ones_shiftVecotr));
	checkCublasErrors(cublasDestroy(handle));
	checkCuFFTErrors(cufftDestroy(plan));
}


/*******************************************
* 函数功能:采用快速傅里叶变换实现相关函数计算
* 输入参数:
* vec_a:     参与相关计算的第一个向量
* vec_b1:    参与相关计算的第二个向量
* vec_Corr： 保存相关函数结果
* Length:    两个向量的长度
* 备注:
* 参与相关计算的两个向量长度是相等的
*******************************************/
void GetCorrelation(thrust::device_vector<float>& vec_a, thrust::device_vector<float>& vec_b1, thrust::device_vector<float>& vec_Corr, int Length)
{
	cufftHandle plan;
	checkCuFFTErrors(cufftPlan1d(&plan, Length, CUFFT_R2C, 1));

	// fft(b1)
	thrust::device_vector<comThr>fft_b1(Length / 2 + 1, comThr(0.0f, 0.0f));
	cuComplex* d_fft_b1 = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(fft_b1.data()));
	float* d_vec_b1 = reinterpret_cast<float*>(thrust::raw_pointer_cast(vec_b1.data()));
	checkCuFFTErrors(cufftExecR2C(plan, d_vec_b1, d_fft_b1));
	// fft(a)
	thrust::device_vector<comThr>fft_a(Length / 2 + 1, comThr(0.0f, 0.0f));
	cuComplex* d_fft_a = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(fft_a.data()));
	float* d_vec_a = reinterpret_cast<float*>(thrust::raw_pointer_cast(vec_a.data()));
	checkCuFFTErrors(cufftExecR2C(plan, d_vec_a, d_fft_a));

	// conj multiply
	thrust::transform(thrust::device, fft_a.begin(), fft_a.end(), fft_b1.begin(), fft_b1.begin(), conjComplex_Mul_Complex());

	// ifft
	float* d_vec_ifft = reinterpret_cast<float*>(thrust::raw_pointer_cast(vec_Corr.data()));
	checkCuFFTErrors(cufftPlan1d(&plan, Length, CUFFT_C2R, 1));
	checkCuFFTErrors(cufftExecC2R(plan, d_fft_b1, d_vec_ifft));

	// fftshift
	thrust::swap_ranges(thrust::device, vec_Corr.begin(), vec_Corr.begin() + Length / 2, vec_Corr.begin() + Length / 2);

	checkCuFFTErrors(cufftDestroy(plan));
}


/*******************************************
* 函数功能:二项式拟合，求最值的精确位置
* 输入参数:
* vec_Corr:  用于求最值的向量
* maxPos:    最大值位置
* 返回参数:
* maxPos:    最大值的精确位置
*******************************************/
float BinomialFix(thrust::device_vector<float>& vec_Corr, unsigned int maxPos)
{
	float f1 = vec_Corr[maxPos - 1];
	float f2 = vec_Corr[maxPos];
	float f3 = vec_Corr[maxPos + 1];

	float fa = (f1 + f3 - 2 * f2) / 2;
	float fb = (f3 - f1) / 2;
	float fc = f2;

	float xstar = -fb / (2 * fa);
	return xstar;
}

/**********************************************
 * 函数功能：将一维像的目标区域移到中间
 * 输入参数：
 * data: 距离像序列，存放在设备上
	   （数据格式：慢时间*快时间，行主序存放）
 * Paras：雷达参数结构体
 **********************************************/
void HRRPCenter(cuComplex* data, RadarParameters paras, const int inter_length)
{
	// * 求平均距离像（最大值归一)
	const int data_length = paras.num_echoes * paras.num_range_bins;
	// 类型转换：cuComplex->thrust
	comThr* thr_data_temp = reinterpret_cast<comThr*>(data);
	thrust::device_ptr<comThr>thr_data = thrust::device_pointer_cast(thr_data_temp);
	// 求包络，并归一化
	thrust::device_vector<float>HRRP0(data_length);
	thrust::transform(thrust::device, thr_data, thr_data + data_length, HRRP0.begin(), Get_abs());
	thrust::device_vector<float>::iterator iter = \
		thrust::max_element(HRRP0.begin(), HRRP0.end());
	float max_abs_hrrp = *iter;
	thrust::transform(thrust::device, HRRP0.begin(), HRRP0.end(), HRRP0.begin(), maxNormalize<float>(max_abs_hrrp));
	// 利用全1向量做乘法，求平均距离像
	thrust::device_vector<float>thr_ones_echo_num(paras.num_echoes, 1.0);
	float* ones_echo_num = reinterpret_cast<float*>(thrust::raw_pointer_cast(thr_ones_echo_num.data()));

	cublasHandle_t handle;
	checkCublasErrors(cublasCreate(&handle));
	float alpha = 1.0 / float(paras.num_echoes);
	float beta = 0.0;
	thrust::device_vector<float>ARP(paras.num_range_bins, 0.0);
	float* d_ARP = (float*)(thrust::raw_pointer_cast(ARP.data()));//*
	float* d_HRRP0 = (float*)(thrust::raw_pointer_cast(HRRP0.data()));

	//checkCublasErrors(cublasSgemv(handle,CUBLAS_OP_T,paras.num_echoes,paras.num_range_bins,&alpha,
	//	                          d_HRRP0,paras.num_echoes,ones_echo_num,1,&beta,d_ARP,1));
	checkCublasErrors(cublasSgemv(handle, CUBLAS_OP_N, paras.num_range_bins, paras.num_echoes, &alpha,
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
	thrust::transform(thrust::device, ARP1.begin() + inter_length, ARP1.end(), ARP1.begin(), diff.begin(), buildDiff(extra_value));
	iter = thrust::min_element(diff.begin(), diff.end());
	unsigned int min_diff_pos = iter - diff.begin();
	float low_threshold_gray = ARP1[min_diff_pos + int(inter_length / 2)];
	// 以下利用thrust实现 indices = find(ARP>low_threshold_gray);
	thrust::device_vector<int>indices(paras.num_range_bins);
	thrust::device_vector<int>::iterator end = thrust::copy_if(thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(int(paras.num_range_bins)),
		ARP.begin(),
		indices.begin(),
		thrust::placeholders::_1 > low_threshold_gray);
	int indices_length = end - indices.begin();
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
	GetARPMean << < grid, block >> > (d_ARP_ave, d_indices, d_ARP, indices_length, WL, paras);

	thrust::device_ptr<float>ARP_ave(d_ARP_ave);

	// 以下实现ind=find(APR_ave<low_threshold_gray); 
	thrust::device_vector<int>ind(indices_length);
	thrust::device_vector<int>::iterator end_min = thrust::copy_if(thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(indices_length),
		ARP_ave,
		ind.begin(),
		thrust::placeholders::_1 < low_threshold_gray);
	int ind_length = end_min - ind.begin();
	ind.resize(ind_length);
	if (ind_length == indices_length)
		return;

	// 使用核函数实现indices(ind)=[];这里直接将索引为ind的置于0
	int* d_ind = (int*)(thrust::raw_pointer_cast(ind.data()));
	block_size = nextPow2(ind_length);    // 块大小扩为离ind_length最近的2的幂
	int set_num = 0;
	setNumInArray<int> << <1, block_size >> > (d_indices, d_ind, set_num, ind_length);

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
	checkCuFFTErrors(cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch));
	checkCuFFTErrors(cufftExecC2C(plan, data, data, CUFFT_FORWARD));

	// 下面利用thrust构造exp(1j*2*pi*Vec*shiftnum) （长度为距离向点数）
	thrust::device_vector<comThr>shift_vector(paras.num_range_bins);
	thrust::device_vector<float>vec(paras.num_range_bins);
	thrust::sequence(thrust::device, vec.begin(), vec.end(), 0.0f);
	thrust::transform(thrust::device, vec.begin(), vec.end(), shift_vector.begin(), buildShiftVec(shift_num, int(paras.num_range_bins)));

	// 通过乘以全一向量实现repmat(exp(1j*2*pi*Vec*shiftnum),n,1)
	thrust::device_vector<comThr>shift_mat(paras.num_echoes * paras.num_range_bins, comThr(0.0f, 0.0f));
	cuComplex* d_shift_mat = (cuComplex*)(thrust::raw_pointer_cast(shift_mat.data()));//*
	thrust::device_vector<comThr>thr_com_ones_echo_num(paras.num_echoes, comThr(1.0f, 0.0f));
	//vectorMulvectorCublasC(handle, d_shift_mat, thr_com_ones_echo_num, shift_vector, paras.num_echoes, paras.num_range_bins);
	vectorMulvectorCublasC(handle, d_shift_mat, shift_vector, thr_com_ones_echo_num, paras.num_range_bins, paras.num_echoes);

	// 利用thrust实现点乘
	thrust::transform(thrust::device, shift_mat.begin(), shift_mat.end(), thr_data, thr_data, Complex_Mul_Complex());

	// ifft
	checkCuFFTErrors(cufftExecC2C(plan, data, data, CUFFT_INVERSE));
	thrust::transform(thrust::device, thr_data, thr_data + data_length, thr_data, ifftNor(paras.num_range_bins));

	// * 释放空间
	checkCublasErrors(cublasDestroy(handle));
	checkCudaErrors(cudaFree(d_ARP_ave));
	checkCuFFTErrors(cufftDestroy(plan));
}

/*************************************************************
 * 函数功能：GPU核函数，根据索引值附近ARP均值剔除野值
 * 参考：    MATLAB程序 HRRPCenter.m
 * 输入：
 *      ARP_ave:  待计算的ARP均值向量，长度为indices_length；
 *      indices:  indices = find(ARP>low_threshold_gray);
 *      ARP：     平均距离像，长度为距离向点数；
 *      WL:       类似于CFAR中的参考单元；
 *      paras:    雷达系统参数
 * 备注：性能有待优化
 *************************************************************/
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

