#ifndef PHASE_ADJUSTMENT_H_
#define PHASE_ADJUSTMENT_H_

#include "common.cuh"


/// <summary>
/// Expand indexing.
/// Expanding the unmber of every element in vector starting from first2 to the corresponding value in vector starting from frist1 and ending at end1.
/// first {2,2,2}. second {1,2,3}. output {1,1,2,2,3,3}.
/// </summary>
/// <typeparam name="InputIterator1"></typeparam>
/// <typeparam name="InputIterator2"></typeparam>
/// <typeparam name="OutputIterator"></typeparam>
/// <param name="first1"></param>
/// <param name="last1"></param>
/// <param name="first2"></param>
/// <param name="output"></param>
/// <returns></returns>
template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator expand(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator output)
{
	typedef typename thrust::iterator_difference<InputIterator1>::type difference_type;

	difference_type input_size = thrust::distance(first1, last1);
	difference_type output_size = thrust::reduce(first1, last1);

	// scan the counts to obtain output offsets for each input element
	thrust::device_vector<difference_type> output_offsets(input_size, 0);
	thrust::exclusive_scan(first1, last1, output_offsets.begin());

	// scatter the nonzero counts into their corresponding output positions
	thrust::device_vector<difference_type> output_indices(output_size, 0);
	thrust::scatter_if (thrust::counting_iterator<difference_type>(0), thrust::counting_iterator<difference_type>(input_size), output_offsets.begin(), first1, output_indices.begin());

	// compute max-scan over the output indices, filling in the holes
	thrust::inclusive_scan(output_indices.begin(), output_indices.end(), output_indices.begin(), thrust::maximum<difference_type>());

	// gather input values according to index array (output = first2[output_indices])
	OutputIterator output_end = output; thrust::advance(output_end, output_size);
	thrust::gather(output_indices.begin(), output_indices.end(), first2, output);

	// return output + output_size
	thrust::advance(output, output_size);
	return output;
}


/// <summary>
/// Doppler tracking. Achieving Coarse phase caliberation.
/// </summary>
/// <param name="d_data"></param>
/// <param name="echo_num"></param>
/// <param name="range_num"></param>
void dopplerTracking(cuComplex* d_data, const int& echo_num, const int& range_num);


/// <summary>
/// d_res = d_data * diag(d_vec)
/// </summary>
/// <param name="d_res"></param>
/// <param name="d_vec"></param>
/// <param name="d_data"></param>
/// <param name="rows"></param>
/// <param name="cols"></param>
/// <returns></returns>
__global__ void Compensate_Phase(cuComplex* d_res, cuComplex* d_vec, cuComplex* d_data, int rows, int cols);


/// <summary>
/// 补偿大带宽下，因大转角引起的随距离空变的二次相位误差
/// </summary>
/// <param name="d_data"></param>
/// <param name="paras"></param>
/// <param name="h_azimuth"></param>
/// <param name="h_pitch"></param>
/// <param name="handle"></param>
void rangeVariantPhaseComp(cuComplex* d_data, const RadarParameters& paras, float* h_azimuth, float* h_pitch, cublasHandle_t handle);


/// <summary>
/// 最小熵快速自聚焦. 参考：邱晓辉《ISAR成像快速最小熵相位补偿方法》，电子与信息学报，2004
/// </summary>
/// <param name="d_data"></param>
/// <param name="echo_num"></param>
/// <param name="range_num"></param>
/// <param name="handle"></param>
void fastEntropy(cuComplex* d_data, const int& echo_num, const int& range_num, cublasHandle_t handle);


/// <summary>
/// 最小熵快速自聚焦中提取距离单元重建回波. 参考：邱晓辉《ISAR成像快速最小熵相位补偿方法》，电子与信息学报，2004
/// </summary>
/// <param name="newData"> 提取距离单元后的回波 </param>
/// <param name="d_Data"> 距离像序列(方位*距离，列主序） </param>
/// <param name="select_bin"> 选出距离单元的索引序列 </param>
/// <param name="NumEcho"> 回波数 </param>
/// <param name="NumRange"> 距离向采样点数 </param>
/// <param name="num_unit2"> 选出的距离单元个数 </param>
/// <returns></returns>
__global__ void Select_Rangebins(cuComplex* newData, cuComplex* d_Data, int* select_bin, int NumEcho, int NumRange, int num_unit2);


/// <summary>
/// realize Doppler centroid tracking to autofocus
/// </summary>
/// <param name="d_preComData"></param>
/// <param name="d_Data"> Range profile(slow-time * fast-time, row major) </param>
/// <param name="NumEcho"></param>
/// <param name="NumRange"></param>
/// <param name="phaseC"></param>
/// <param name="isNeedCompensation"></param>
void dopplerTracking2(cuComplex* d_preComData, cuComplex* d_data, const int& echo_num, const int& range_num, thrust::device_vector<comThr>& phaseC, int isNeedCompensation);

#endif // !PHASE_ADJUSTMENT_H_
