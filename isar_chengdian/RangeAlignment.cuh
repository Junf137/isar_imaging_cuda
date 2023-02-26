#ifndef RANGEALIGNMENT_H_
#define RANGEALIGNMENT_H_

#include <cstdlib>
#include <complex>
#include <iostream>
#include <cstdio>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"
#include "cublas_v2.h"
#include "cufft.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/complex.h>
#include <thrust/scan.h>
#include <thrust/swap.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include "Common.cuh"

/************************
* thrust库所用functor
*************************/

// ======== 构建频谱中心化向量 ======== //
struct GenerateShiftVector : public thrust::binary_function<float, float, comThr>
{
	__host__ __device__
		comThr operator()(float x, float c) {
		return (x * thrust::exp(comThr(0.0f, -PI * c)));
	}
};

// ======== ifft数据除以N后取绝对值 ======== //
struct ifftNor_abs : public thrust::unary_function<comThr, float>
{
	const int nfft;
	ifftNor_abs(int _nfft) : nfft(_nfft) {}
	__host__ __device__
		float operator() (comThr& x) const {
		return thrust::abs(x / float(nfft));
	}
};

// ======== 构造频移向量 ======== //
struct Fre_shift : public thrust::unary_function<float, comThr>
{
	const int N;
	const float mopt;
	Fre_shift(int _N, float _mopt) : N(_N), mopt(_mopt) {}
	__host__ __device__
		comThr operator() (float& x) const {
		return thrust::exp(comThr(0.0f, -2 * PI * x * mopt / N));
	}
};

// ======== 用已经对齐的回波更新模板 ======== //
struct Renew_b1 : public thrust::binary_function<float, comThr, float>
{
	__host__ __device__
		float operator() (float& x, comThr& y) const {
		return (x * 0.95f + thrust::abs(y));
	}
};

// ======== 构造中心化函数中的diff向量 ======== //
struct buildDiff : public thrust::binary_function<float, float, float>
{
	const float extra_value;
	buildDiff(float _extra_value) : extra_value(_extra_value) {};
	__host__ __device__
		float operator()(float& x, float& y) const {
		return std::abs(((x - y - extra_value)));
	}
};

// ======== 构造中心化函数中的indices ======== //
struct isLargerThanThreshold
{
	float threshold;
	isLargerThanThreshold(float _threshold) : threshold(_threshold) {};
	__host__ __device__
		int operator()(float x) {
		return (x > threshold) ? 1 : 0;
	}
};

// ======== 构造HRRPCenter最后的平移向量 ======== //
struct buildShiftVec :public thrust::unary_function<float, comThr>
{
	const int shift_num;
	const int m;
	buildShiftVec(int _shift_num, int _m) : shift_num(_shift_num), m(_m) {};
	__host__ __device__
		comThr operator()(float& x) const {
		return (thrust::exp(comThr(0.0, 2 * PI * x * float(shift_num) / float(m))));
	}
};
/***********
* 函数声明
************/

// linej算法实现包络对齐
void RangeAlignment_linej(cuComplex* data, RadarParameters paras, thrust::device_vector<int>& shift_vec);

// 获取相关函数
void GetCorrelation(thrust::device_vector<float>& vec_a, thrust::device_vector<float>& vec_b1, thrust::device_vector<float>& vec_Corr, int Length);

// 二项拟合
float BinomialFix(thrust::device_vector<float>& vec_Corr, unsigned int maxPos);

// 将距离像序列平移到画面中心
void HRRPCenter(cuComplex* data, RadarParameters paras, const int inter_length);

// 剔除野值
__global__ void GetARPMean(float* ARP_ave, int* indices, float* ARP, int indices_length, int WL, RadarParameters paras);

#endif // !RANGEALIGNMENT_H_
