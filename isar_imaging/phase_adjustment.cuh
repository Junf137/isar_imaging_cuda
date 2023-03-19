#ifndef PHASE_ADJUSTMENT_H_
#define PHASE_ADJUSTMENT_H_

#include "common.cuh"


/// <summary>
/// Doppler tracking. Achieving Coarse phase caliberation.
/// </summary>
/// <param name="d_data"></param>
/// <param name="echo_num"></param>
/// <param name="range_num"></param>
void dopplerTracking(cuComplex* d_data, const int& echo_num, const int& range_num);


/// <summary>
/// d_res = diag(d_diag) * d_data
/// d_diag is a vector of size len / cols.
/// </summary>
/// <param name="d_diag"></param>
/// <param name="d_data"></param>
/// <param name="d_res"></param>
/// <param name="cols"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void diagMulMat(cuComplex* d_diag, cuComplex* d_data, cuComplex* d_res, int cols, int len);


/// <summary>
/// 补偿大带宽下，因大转角引起的随距离空变的二次相位误差
/// </summary>
/// <param name="d_data"></param>
/// <param name="paras"></param>
/// <param name="h_azimuth"></param>
/// <param name="h_pitch"></param>
/// <param name="handle"></param>
void rangeVariantPhaseComp(cuComplex* d_data, float* h_azimuth, float* h_pitch, const RadarParameters& paras, const CUDAHandle& handles);


/// <summary>
/// 最小熵快速自聚焦. 参考：邱晓辉《ISAR成像快速最小熵相位补偿方法》，电子与信息学报，2004
/// </summary>
/// <param name="d_data"></param>
/// <param name="echo_num"></param>
/// <param name="range_num"></param>
/// <param name="handle"></param>
void fastEntropy(cuComplex* d_data, const int& echo_num, const int& range_num, const CUDAHandle& handles);


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
