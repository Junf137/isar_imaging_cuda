#ifndef RVPCOMPENSATION_H_
#define RVPCOMPENSATION_H_

#include <cstdlib>
#include <complex>
#include <iostream>
#include <cstdio>

#include "cuda_runtime.h"
#include "cuComplex.h"
#include "cublas_v2.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/complex.h>
#include <thrust/scan.h>
#include <thrust/swap.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>

#include "Common.cuh"


/**************************
* thrust库所用functor
**************************/

// ======== 建立频率向量乘以df ======== //
struct mul_by_df :public thrust::unary_function<float, float>
{
	float df;
	mul_by_df(float _df) : df(_df) {}
	__host__ __device__
		float operator()(float& x) const {
		return x * df;
	}
};

// ======== 构造RVP消除项 ======== //
struct build_rvp_compensation :public thrust::unary_function<float, comThr>
{
	float chirp_rate;
	build_rvp_compensation(float _chirp_rate) :chirp_rate(_chirp_rate) {}
	__host__ __device__
		comThr operator()(float& x) const {
		return (thrust::exp(comThr(0.0, -PI * x * x / chirp_rate)));
	}
};

/**************
* 函数声明
***************/

// 实现RVP消除
void RVPCompensation(cuComplex* d_data, RadarParameters paras, cublasHandle_t handle_RVPC);


#endif // !RVPCOMPENSATION_H_
