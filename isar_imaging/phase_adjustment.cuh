#ifndef PHASE_ADJUSTMENT_H_
#define PHASE_ADJUSTMENT_H_

#include "common.cuh"


/// <summary>
/// Doppler tracking. Achieving Coarse phase calibration.
/// </summary>
/// <param name="d_data_comp"></param>
/// <param name="d_phase"></param>
/// <param name="d_data"></param>
/// <param name="echo_num"></param>
/// <param name="range_num"></param>
/// <param name="if_compensation"></param>
void dopplerTracking(cuComplex* d_data_comp, cuComplex* d_phase, cuComplex* d_data, const int& echo_num, const int& range_num, const bool& if_compensation);


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
/// compensating the second-order phase error caused by large angle of rotation in large bandwidth
/// </summary>
/// <param name="d_data"></param>
/// <param name="paras"></param>
/// <param name="h_azimuth"></param>
/// <param name="h_pitch"></param>
/// <param name="handle"></param>
void rangeVariantPhaseComp(cuComplex* d_data, float* h_azimuth, float* h_pitch, const RadarParameters& paras, const CUDAHandle& handles);


/// <summary>
/// fast minimum entropy auto-focus. (Reference: 邱晓辉《ISAR成像快速最小熵相位补偿方法》，电子与信息学报，2004)
/// </summary>
/// <param name="d_data"></param>
/// <param name="echo_num"></param>
/// <param name="range_num"></param>
/// <param name="handle"></param>
void fastEntropy(cuComplex*& d_data, const int& echo_num, const int& range_num, const CUDAHandle& handles);


/// <summary>
/// Rebuild echo after extracting range bins of selected index in fast entropy auto-focus. (Reference: 邱晓辉《ISAR成像快速最小熵相位补偿方法》，电子与信息学报，2004)
/// Each block is responsible for extracting a single row, and each thread is responsible for extracting a single element inside a row.
/// Kernel configuration requirements:
/// (1) block_number == echo_num
/// (2) thread_per_block == num_unit2
/// </summary>
/// <param name="d_new_data"> new data of extracted range bins </param>
/// <param name="d_data"></param>
/// <param name="select_bin"> index of selected range data </param>
/// <param name="echo_num"></param>
/// <param name="range_num"></param>
/// <param name="num_unit2"> number of selected range data </param>
/// <returns></returns>
__global__ void selectRangeBins(cuComplex* d_new_data, cuComplex* d_data, int* select_bin, int echo_num, int range_num, int num_unit2);


#endif // !PHASE_ADJUSTMENT_H_
