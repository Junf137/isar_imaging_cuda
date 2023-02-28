#ifndef RANGEALIGNMENT_H_
#define RANGEALIGNMENT_H_

#include "common.cuh"


/// <summary>
/// Range alignment using linej algorithm
/// </summary>
/// <param name="d_data"></param>
/// <param name="hamming_window"></param>
/// <param name="paras"></param>
/// <param name="handle"></param>
/// <param name="plan_one_echo_c2c"></param>
/// <param name="plan_one_echo_r2c"></param>
/// <param name="plan_one_echo_c2r"></param>
void rangeAlignment(cuComplex* d_data, float* hamming_window, RadarParameters paras, cublasHandle_t handle, cufftHandle plan_one_echo_c2c, cufftHandle plan_one_echo_r2c, cufftHandle plan_one_echo_c2r);


/// <summary>
/// hamming .* real(exp(-1j*pi*[0:N-1]))
/// </summary>
/// <param name="hamming"></param>
/// <param name="d_freq_centering_vec"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void genFreqCenteringVec(float* hamming, cuComplex* d_freq_centering_vec, int len);


/// <summary>
/// Calculate corelation function using fft algorithm.
/// Two engaged vectors are required to share the same length.
/// </summary>
/// <param name="vec_a"> first vector </param>
/// <param name="vec_b1"> second vector </param>
/// <param name="vec_Corr"> correlation result </param>
void getCorrelation(float* d_vec_corr, float* d_vec_a, float* d_vec_b, int len, cufftHandle plan_one_echo_r2c, cufftHandle plan_one_echo_c2r);


/// <summary>
/// Binomial fixing, get the precise position of max value.
/// </summary>
/// <param name="d_vec_corr"></param>
/// <param name="d_xstar"></param>
/// <param name="maxPos"></param>
/// <returns></returns>
__global__ void binomialFix(float* d_vec_corr, float* d_xstar, int maxPos);


/// <summary>
/// Updating d_vec_b in range alignment iteration.
/// d_vec_b = 0.95f * d_vec_b + abs(d_data(i,:))
/// </summary>
/// <param name="d_vec_b"></param>
/// <param name="d_data_i"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void updateVecB(float* d_vec_b, cuComplex* d_data_i, int len);


/// <summary>
/// d_freq_mov_vec = exp(-1j * 2 * pi * [0:N-1] * mopt / N)
/// </summary>
/// <param name="d_freq_mov_vec"></param>
/// <param name="mopt"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void genFreqMovVec(cuComplex* d_freq_mov_vec, float mopt, int len);


/// <summary>
/// Centering HRRP
/// </summary>
/// <param name="data"> 距离像序列，存放在设备上（数据格式：慢时间* 快时间，行主序存放） </param>
/// <param name="paras">  </param>
/// <param name="inter_length">  </param>
/// <param name="handle">  </param>
void HRRPCenter(cuComplex* d_data, RadarParameters paras, const int inter_length, cublasHandle_t handle, cufftHandle plan_all_echo_c2c);

/// <summary>
/// GPU核函数，根据索引值附近ARP均值剔除野值. (参考: HRRPCenter.m)
/// 备注: 性能有待优化
/// </summary>
/// <param name="ARP_ave"> 待计算的ARP均值向量，长度为indices_length </param>
/// <param name="indices"> indices = find(ARP>low_threshold_gray) </param>
/// <param name="ARP"> 平均距离像，长度为距离向点数 </param>
/// <param name="indices_length"></param>
/// <param name="WL"> 类似于CFAR中的参考单元 </param>
/// <param name="paras"> 雷达系统参数 </param>
/// <returns></returns>
__global__ void GetARPMean(float* ARP_ave, int* indices, float* ARP, int indices_length, int WL, RadarParameters paras);

#endif // !RANGEALIGNMENT_H_
