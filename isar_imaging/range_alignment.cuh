#ifndef RANGEALIGNMENT_H_
#define RANGEALIGNMENT_H_

#include "common.cuh"


/// <summary>
/// 采用linej算法的距离对准函数, 参考程序:juliduizhun_modified2.m
/// </summary>
/// <param name="data"> 距离像序列，存放在设备上（数据格式：慢时间* 快时间，行主序存放） </param>
/// <param name="paras"> 雷达参数结构体 </param>
/// <param name="shift_vec"> fftshift vector </param>
/// <param name="handle">  </param>
void RangeAlignment_linej(cuComplex* d_data, RadarParameters paras, thrust::device_vector<int>& fftshift_vec, cublasHandle_t handle);


/// <summary>
/// Calculate corelation function using fft algorithm.
/// Two engaged vectors are required to share the same length.
/// </summary>
/// <param name="vec_a"> first vector </param>
/// <param name="vec_b1"> second vector </param>
/// <param name="vec_Corr"> corelation result </param>
void GetCorrelation(thrust::device_vector<float>& vec_a, thrust::device_vector<float>& vec_b1, thrust::device_vector<float>& vec_Corr);


/// <summary>
/// 二项式拟合，求最值的精确位置
/// </summary>
/// <param name="vec_Corr"> 用于求最值的向量 </param>
/// <param name="maxPos"> 最大值位置 </param>
/// <returns></returns>
float BinomialFix(thrust::device_vector<float>& vec_Corr, int maxPos);


/// <summary>
/// 将一维像的目标区域移到中间
/// </summary>
/// <param name="data"> 距离像序列，存放在设备上（数据格式：慢时间* 快时间，行主序存放） </param>
/// <param name="paras"> 雷达参数结构体 </param>
/// <param name="inter_length">  </param>
/// <param name="handle">  </param>
void HRRPCenter(cuComplex* data, RadarParameters paras, const int inter_length, cublasHandle_t handle);

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
