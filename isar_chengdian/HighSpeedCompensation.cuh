#ifndef HIGHSPEEDCOMPENSATION_H_
#define HIGHSPEEDCOMPENSATION_H_

#include "Common.cuh"
#include "cuComplex.h"
#include "cublas_v2.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/complex.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

/********************************
 * thrust库所用functor
 ********************************/

 // ======== 建立快时间向量除以fs ======== //
struct nor_by_fs :public thrust::unary_function<float, float>
{
	unsigned int Fs;
	nor_by_fs(unsigned int _Fs) : Fs(_Fs) {}
	__host__ __device__
		float operator()(float& x) const {
		return x / float(Fs);
	}
};

// ======== 实现4个thrust向量累加 ======== //
struct arbitrary_functor_four_sum
{
	template<typename Tuple>
	__host__ __device__
		void operator()(Tuple t)
	{
		thrust::get<4>(t) = thrust::get<3>(t) + thrust::get<2>(t) + thrust::get<1>(t) + thrust::get<0>(t);
	}
};

// ======== 求相位的指数 ======== //
struct getPhi_functor :public thrust::unary_function<float, comThr>
{
	__host__ __device__
		comThr operator() (float& x) const {
		return thrust::exp(comThr(0.0, x));
	}
};

/*************************
* 函数声明
* 用以实现高速运动补偿
**************************/
void HighSpeedCompensation(cuComplex* d_data, unsigned int Fs, long long fc, long long band_width, float Tp, float* velocity_data,
	float* range_data, int echo_num, int range_num, cublasHandle_t handle_HSC);


#endif /* HIGHSPEEDCOMPENSATION_H_ */