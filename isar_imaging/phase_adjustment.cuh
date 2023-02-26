#ifndef PHASE_ADJUSTMENT_H_
#define PHASE_ADJUSTMENT_H_

#include "common.cuh"

/*******************
 * thrust所需结构体
 *******************/
struct Get_Angle :public thrust::unary_function<comThr, float>
{
	__host__ __device__
		float operator()(comThr x)
	{

		return thrust::arg((x / thrust::abs(x)));
	}
};

struct Get_Com_Phase :public thrust::unary_function<float, comThr>
{
	__host__ __device__
		comThr operator()(float x)
	{

		return thrust::exp(comThr(0.0, -x));
	}
};

struct BuildRangeVector :public thrust::unary_function<float, float>
{
	const float resolution;
	const float wave_length;
	BuildRangeVector(float _resolution, float _wave_length) :resolution(_resolution), wave_length(_wave_length) {};
	__host__ __device__
		float operator()(float x)
	{
		return x * 2.0f * resolution / wave_length;
	}
};

struct GetAngle :public thrust::binary_function<float, float, float>
{
	const float mid_x;
	const float mid_y;
	const float mid_z;
	GetAngle(float _mid_x, float _mid_y, float _mid_z) : mid_x(_mid_x), mid_y(_mid_y), mid_z(_mid_z) {};
	__host__ __device__
		float operator()(float cur_azi, float cur_pit)
	{
		float x = sinf(cur_pit / 180 * PI_h);
		float y = cosf(cur_pit / 180 * PI_h) * cosf(cur_azi / 180 * PI_h);
		float z = cosf(cur_pit / 180 * PI_h) * sinf(cur_azi / 180 * PI_h);

		float angle = (x * mid_x + y * mid_y + z * mid_z);
		angle = acosf(angle);
		float angle2;
		angle2 = powf(angle, 2.0);
		return angle2;
	}
};

struct CompensatePhase :public thrust::binary_function<float, comThr, comThr>
{
	__host__ __device__
		comThr operator()(float x, comThr y)
	{
		return y * thrust::exp(comThr(0.0, -2 * PI_h * x));
	}
};

struct isWithinThreshold
{
	const float threshold;
	isWithinThreshold(float _threshold) : threshold(_threshold) {}
	__host__ __device__
		bool operator()(float x)const {
		return (x > threshold);
	}
};

struct Normlize :public thrust::unary_function<float, float>
{
	int N;
	Normlize(int _N) : N(_N) {}
	__host__ __device__
		float operator()(float x)const {
		return x / float(N);
	}
};

struct absSquare :public thrust::unary_function<comThr, float>
{
	__host__ __device__
		float operator()(comThr& x)const {
		return powf(thrust::abs(x), 2);
	}
};

struct GetEntropy :public thrust::unary_function<float, float>
{
	__host__ __device__
		float operator()(float& x)const {
		return -(x * logf(x));
	}
};

struct Complex_mul_absComplex :public thrust::unary_function<comThr, comThr>
{
	__host__ __device__
		comThr operator()(comThr& x)const {
		return x * thrust::abs(x);
	}
};

struct Elementwise_normalize :public thrust::unary_function<comThr, comThr>
{
	__host__ __device__
		comThr operator()(comThr x)
	{
		return (x / thrust::abs(x));
	}
};

struct Conj_mul_ifftNor
{
	const int N;
	Conj_mul_ifftNor(int _N) : N(_N) {};
	__host__ __device__
		comThr operator()(comThr& x, comThr& y)const {
		return (thrust::conj(x) * (y / float(N)));
	}
};

/**********************************
 * 模板函数定义
 * 该函数用于扩展索引
 * 例如：
 * 第一个输入的迭代器是：{2,2,2}
 * 第二个输入的迭代器是：{1,2,3}
 * 输出为：{1,1,2,2,3,3}
 **********************************/
template <typename InputIterator1,
	typename InputIterator2,
	typename OutputIterator>
OutputIterator expand(InputIterator1 first1,
	InputIterator1 last1,
	InputIterator2 first2,
	OutputIterator output)
{
	typedef typename thrust::iterator_difference<InputIterator1>::type difference_type;

	difference_type input_size = thrust::distance(first1, last1);
	difference_type output_size = thrust::reduce(first1, last1);

	// scan the counts to obtain output offsets for each input element
	thrust::device_vector<difference_type> output_offsets(input_size, 0);
	thrust::exclusive_scan(first1, last1, output_offsets.begin());

	// scatter the nonzero counts into their corresponding output positions
	thrust::device_vector<difference_type> output_indices(output_size, 0);
	thrust::scatter_if
	(thrust::counting_iterator<difference_type>(0),
		thrust::counting_iterator<difference_type>(input_size),
		output_offsets.begin(),
		first1,
		output_indices.begin());

	// compute max-scan over the output indices, filling in the holes
	thrust::inclusive_scan
	(output_indices.begin(),
		output_indices.end(),
		output_indices.begin(),
		thrust::maximum<difference_type>());

	// gather input values according to index array (output = first2[output_indices])
	OutputIterator output_end = output; thrust::advance(output_end, output_size);
	thrust::gather(output_indices.begin(),
		output_indices.end(),
		first2,
		output);

	// return output + output_size
	thrust::advance(output, output_size);
	return output;
}

/**************
 * 函数声明   *
 **************/

 // 多普勒跟踪粗校准
void Doppler_Tracking(cuComplex* d_data, RadarParameters paras);

// 用以多普勒跟踪时的相位粗补偿
__global__ void Compensate_Phase(cuComplex* d_res, cuComplex* d_vec, cuComplex* d_data, int rows, int cols);

// 补偿大带宽下因距离空变引起的二次相位误差
void RangeVariantPhaseComp(cuComplex* d_data, RadarParameters paras, float* azimuth_data, float* pitch_data);

// 快速最小熵相位补偿算法
void Fast_Entropy(cuComplex* d_Data, RadarParameters paras);

// 快速最小熵相位补偿算法中的距离单元提取
__global__ void Select_Rangebins(cuComplex* newData, cuComplex* d_Data, int* select_bin, int NumEcho, int NumRange, int num_unit2);

// 另一版本的多普勒跟踪粗补偿
void Doppler_Tracking2(cuComplex* d_preComData, cuComplex* d_Data, int NumEcho, int NumRange, thrust::device_vector<comThr>& phaseC, int isNeedCompensation);

#endif // !PHASE_ADJUSTMENT_H_
