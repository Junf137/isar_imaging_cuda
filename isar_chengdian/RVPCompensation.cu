#include "RVPCompensation.cuh"
#include "Common.cuh"


/******************************************
 * 函数功能：对去斜数据实现RVP消除
 * 输入参数：
 * d_data:   距离像序列，按回波依次存入内存中
 * paras:    雷达参数信息
 * 备注：    参考MALAB程序以及
 *           宽带雷达STRETCH处理系统失真补偿新方法_林钱强
 ******************************************/
void RVPCompensation(cuComplex* d_data, RadarParameters paras, cublasHandle_t handle_RVPC)
{
	float frequency_interval = float(paras.Fs) / float(paras.num_range_bins - 1);    // df = fs / (range_num-1)

	float frequency_start = -float(paras.num_range_bins) / 2.0;                      // fi = (-range_num/2:range_num/2-1)*df
	thrust::device_vector<float>frequency(paras.num_range_bins);
	thrust::sequence(thrust::device, frequency.begin(), frequency.end(), frequency_start);
	thrust::transform(thrust::device, frequency.begin(), frequency.end(), frequency.begin(), mul_by_df(frequency_interval));

	float chirp_rate = float(paras.band_width) / paras.Tp;

	thrust::device_vector<comThr>RVP_compensation_term(paras.num_range_bins);        // Sc = exp(-1j*pi*fi^2/K)
	thrust::transform(thrust::device, frequency.begin(), frequency.end(), RVP_compensation_term.begin(), build_rvp_compensation(chirp_rate));

	thrust::device_vector<comThr>all_ones_256(paras.num_echoes);                     // 构造全1向量
	thrust::fill(thrust::device, all_ones_256.begin(), all_ones_256.end(), comThr(1.0, 0.0));

	// 类型转换：thrust->cuComplex
	cuComplex* d_RVP_compensation_matrix;
	checkCudaErrors(cudaMalloc((void**)&d_RVP_compensation_matrix, sizeof(cuComplex) * paras.num_echoes * paras.num_range_bins));
	checkCudaErrors(cudaMemset(d_RVP_compensation_matrix, 0.0, sizeof(cuComplex) * paras.num_echoes * paras.num_range_bins));

	// 构造RVP补偿矩阵
	//vectorMulvectorCublasC(handle_RVPC, d_RVP_compensation_matrix, all_ones_256, RVP_compensation_term,paras.num_echoes,paras.num_range_bins);
	vectorMulvectorCublasC(handle_RVPC, d_RVP_compensation_matrix, RVP_compensation_term, all_ones_256, paras.num_range_bins, paras.num_echoes);

	comThr* thr_data_temp = reinterpret_cast<comThr*>(d_data);                      // 类型转换：cuComplex->thrust
	thrust::device_ptr<comThr>thr_data = thrust::device_pointer_cast(thr_data_temp);

	comThr* thr_RVP_compensation_matrix_temp = reinterpret_cast<comThr*>(d_RVP_compensation_matrix);
	thrust::device_ptr<comThr>thr_RVP_compensation_matrix = thrust::device_pointer_cast(thr_RVP_compensation_matrix_temp);

	thrust::transform(thrust::device, thr_data, thr_data + paras.num_echoes * paras.num_range_bins,
		thr_RVP_compensation_matrix, thr_data, Complex_Mul_Complex());              // RVP消除

	checkCudaErrors(cudaFree(d_RVP_compensation_matrix));
	//checkCublasErrors(cublasDestroy(handle_RVPC));
}