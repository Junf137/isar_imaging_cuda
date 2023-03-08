#ifndef COMMON_H_
#define COMMON_H_

//#define DATA_WRITE_BACK

// cuda runtime
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <cufft.h>
#include "helper_cuda.h"

// thrust lib
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>	
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/gather.h>

// stl lib
#include <complex>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>

typedef std::vector<std::vector<float>> vec2D_FLOAT;
typedef std::vector<std::vector<int>> vec2D_INT;
typedef std::vector<std::complex<float>> vec1D_COM_FLOAT;
typedef std::vector<std::vector<std::complex<float>>> vec2D_COM_FLOAT;
typedef thrust::complex<float> comThr;


constexpr const char* DIR_PATH = "F:\\Users\\Project\\isar_imaging\\210425235341_047414_1383_00\\";
constexpr auto PI_h = 3.14159265358979f;
constexpr auto LIGHT_SPEED_h = 300000000;

namespace fs = std::filesystem;


struct RadarParameters
{
	int echo_num;
	int range_num;
	long long band_width;
	long long fc;
	float Tp;
	int Fs;
};


/// <summary>
/// result_matrix = alpah * vec1 * vec2.' + result_matrix;
/// for cuComplex number.
/// </summary>
/// <param name="handle"> cublas handle </param>
/// <param name="result_matrix"> multiply result matrix of size m * n (store in column major) </param>
/// <param name="vec1">  vector of length m </param>
/// <param name="vec2"> vecto of length n </param>
/// <param name="alpha"> alpha can be in host or device memory </param>
void vecMulvec(cublasHandle_t handle, cuComplex* result_matrix, thrust::device_vector<comThr>& vec1, thrust::device_vector<comThr>& vec2, const cuComplex& alpha);  // todo: deprecated
void vecMulvec(cublasHandle_t handle, cuComplex* d_vec1, int len1, cuComplex* d_vec2, int len2, cuComplex* d_res_matrix, const cuComplex& alpha);


/// <summary>
/// result_matrix = alpah * vec1 * vec2.' + result_matrix;
/// for float number.
/// </summary>
/// <param name="handle"> cublas handle </param>
/// <param name="result_matrix"> multiply result matrix of size m * n (store in column major) </param>
/// <param name="vec1">  vector of length m </param>
/// <param name="vec2"> vecto of length n </param>
/// <param name="alpha"> alpha can be in host or device memory </param>
void vecMulvec(cublasHandle_t handle, float* result_matrix, thrust::device_vector<float>& vec1, thrust::device_vector<float>& vec2, const float& alpha);  // todo: deprecated
void vecMulvec(cublasHandle_t handle, float* d_vec1, int len1, float* d_vec2, int len2, float* d_res_matrix, const float& alpha);

/// <summary>
/// 
/// </summary>
/// <param name="handle"></param>
/// <param name="d_vec"></param>
/// <param name="len"></param>
/// <param name="max_idx"></param>
/// <param name="max"></param>
void getMax(cublasHandle_t handle, float* d_vec, int len, int* h_max_idx, float* h_max_val);


/// <summary>
/// 
/// </summary>
/// <param name="handle"></param>
/// <param name="d_vec"></param>
/// <param name="len"></param>
/// <param name="h_max_idx"></param>
/// <param name="h_max_val"></param>
void getMax(cublasHandle_t handle, cuComplex* d_vec, int len, int* h_max_idx, cuComplex* h_max_val);


/// <summary>
/// 
/// </summary>
/// <param name="handle"></param>
/// <param name="d_vec"></param>
/// <param name="len"></param>
/// <param name="min_idx"></param>
/// <param name="min_val"></param>
void getMin(cublasHandle_t handle, float* d_vec, int len, int* min_idx, float* min_val);


__global__ void elementwiseAbs(cuComplex* a, float* abs, int len);


/// <summary>
/// Perform element-wise vector multiplcation.
/// c = a .* b
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="c"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void elementwiseMultiply(cuComplex* a, cuComplex* b, cuComplex* c, int len);


/// <summary>
/// Perform element-wise vector multiplcation.
/// c = conj(a) .* b
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="c"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void elementwiseMultiplyConjA(cuComplex* a, cuComplex* b, cuComplex* c, int len);


/// <summary>
/// 
/// </summary>
/// <param name="a">vector with length len_a</param>
/// <param name="b">vector with length len_b</param>
/// <param name="c">vector with length len_c = len_b</param>
/// <param name="len_a"></param>
/// <param name="len_b"></param>
/// <returns></returns>
__global__ void elementwiseMultiplyRep(cuComplex* a, cuComplex* b, cuComplex* c, int len_a, int len_b);


/// <summary>
/// Perform element-wise vector multiplcation.
/// c = a .* b
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="c"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void elementwiseMultiply(float* a, float* b, float* c, int len);


/// <summary>
/// Perform element-wise vector multiplcation.
/// c = a .* b
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="c"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void elementwiseMultiply(float* a, cuComplex* b, cuComplex* c, int len);


/// <summary>
/// 
/// </summary>
/// <param name="a">vector with length len_a</param>
/// <param name="b">vector with length len_b</param>
/// <param name="c">vector with length len_c = len_b</param>
/// <param name="len_a"></param>
/// <param name="len_b"></param>
/// <returns></returns>
__global__ void elementwiseMultiplyRep(float* a, cuComplex* b, cuComplex* c, int len_a, int len_b);


/// <summary>
/// Perform element-wise vector multiplcation.
/// c = a .* b
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="c"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void elementwiseMultiply(float* a, cuComplex* b, float* c, int len);


/// <summary>
/// res = exp(1j * x)
/// </summary>
/// <param name="x"></param>
/// <param name="res"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void expJ(float* x, cuComplex* res, int len);


/// <summary>
/// Generating Hamming Window.
/// </summary>
/// <param name="hamming"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void genHammingVec(float* hamming, int len);


template <typename T>
/// <summary>
/// 
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void swap_range(T* a, T* b, int len)  // todo: separate definition and declaration
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < len) {
		T c = a[tid]; a[tid] = b[tid]; b[tid] = c;
	}
}


//template <typename T>
///// <summary>
///// refference: https://stackoverflow.com/questions/27925979/thrustmax-element-slow-in-comparison-cublasisamax-more-efficient-implementat
///// </summary>
///// <param name="data"></param>
///// <param name="dsize"></param>
///// <param name="result"></param>
///// <returns></returns>
//__global__ void getMaxIdx(const T* data, const int dsize, int* result);


/// <summary>
/// get maximum value of every column
/// Refference:  https://stackoverflow.com/questions/17698969/determining-the-least-element-and-its-position-in-each-matrix-column-with-cuda-t
/// </summary>
/// <param name="c"> data set, column is not coalescing </param>
/// <param name="maxval"> vector to store maximum value </param>
/// <param name="maxidx"> vector to store maximum value's index </param>
/// <param name="row"> number of rows </param>
/// <param name="col"> number of columns ></param>
void getMaxInColumns(thrust::device_vector<float>& c, thrust::device_vector<float>& maxval, thrust::device_vector<int>& maxidx, int row, int col);


/// <summary>
/// Cutting range profile along echo dimension
/// </summary>
/// <param name="d_data"></param>
/// <param name="paras"></param>
/// <param name="range_num_cut"></param>
/// <param name="handle"></param>
void cutRangeProfile(cuComplex*& d_data, RadarParameters& paras, const int& range_num_cut, cublasHandle_t handle);


/// <summary>
/// 实现距离像的径向截取
/// </summary>
/// <param name="d_in"> 原始距离像序列(按回波依次存储, 下同) </param>
/// <param name="d_out"> 截取后的距离像序列 </param>
/// <param name="data_size"> 截取后的距离像序列元素个数(行数*列数) </param>
/// <param name="offset"> 每个回波截取的起始点 </param>
/// <param name="num_elements"> 截取后, 每个回波的点数 </param>
/// <param name="num_ori_elements"> 截取前, 每个回波的点数 </param>
/// <returns></returns>
__global__ void cutRangeProfileHelper(cuComplex* d_in, cuComplex* d_out, const int data_size,
	const int offset, const int num_elements, const int num_ori_elements);


/// <summary>
/// 实现 2^nextpow2(N). MATLAB里是只返回指数, 这里一并求出了2的幂
/// </summary>
/// <param name="N"></param>
/// <returns></returns>
int nextPow2(int N);


/// <summary>
/// 将array中位于index处的元素置为set_num
/// </summary>
/// <param name="arrays"> 数据序列 </param>
/// <param name="index"> 要被替换的元素索引 </param>
/// <param name="set_num"> 替换值 </param>
/// <param name="num_index"> 元素个数 </param>
/// <returns></returns>
__global__ void setNumInArray(int* arrays, int* index, int set_num, int num_index);


/// <summary>
/// HRRP
/// </summary>
/// <param name="plan"></param>
/// <param name="d_hrrp"></param>
/// <param name="d_data"></param>
/// <param name="echo_num"></param>
/// <param name="range_num"></param>
/// <param name="d_fftshift"></param>
void getHRRP(cuComplex* d_hrrp, cuComplex* d_data, const int& echo_num, const int& range_num, float* hamming, cufftHandle plan_all_echo_c2c);


/// <summary>
/// 计算目标旋转角度, 计算两个雷达观测方向的单位矢量
/// </summary>
/// <param name="azimuth1"></param>
/// <param name="pitching1"></param>
/// <param name="azimuth2"></param>
/// <param name="pitching2"></param>
/// <returns></returns>
float getTurnAngle(const float& azimuth1, const float& pitching1, const float& azimuth2, const float& pitching2);


/// <summary>
/// 计算转角变化曲线.
/// 先找出跟踪丢失点, 对每一段估计, 然后再加起来.
/// </summary>
/// <param name="turnAngle"> 目标转动角度曲线 </param>
/// <param name="azimuth"> 方位角 </param>
/// <param name="pitching"> 俯仰角 </param>
/// <returns></returns>
int turnAngleLine(std::vector<float>* turnAngle, const std::vector<float>& azimuth, const std::vector<float>& pitching);


/// <summary>
/// Returns interpolated value at x from parallel arrays ( xData, yData )
/// Assumes that xData has at least two elements, is sorted and is strictly monotonic increasing
/// boolean argument extrapolate determines behaviour beyond ends of array (if needed)
/// </summary>
/// <param name="xData"></param>
/// <param name="yData"></param>
/// <param name="x"> interpolation point </param>
/// <param name="extrapolate"> option for extrapolate </param>
/// <returns></returns>
float interpolate(const std::vector<int>& xData, const std::vector<float>& yData, const int& x, const bool& extrapolate);


// [flag_data_end DataW_FileSn DataNOut TurnAngleOut] = UniformitySampling(DataN, TurnAngle, CQ, WindowHead, WindowLength)
int uniformSamplingFun(int* flagDataEnd, std::vector<int>* dataWFileSn, vec2D_FLOAT* dataNOut, std::vector<float>* turnAngleOut, \
	const vec2D_FLOAT& dataN, const std::vector<float>& turnAngle, const int& CQ, const int& windowHead, const int& windowLength);


//function [flag_data_end DataW_FileSn DataNOut TurnAngleOut] = NonUniformitySampling(DataN, RadarParameters, TurnAngle, start, M)
int nonUniformSamplingFun();


/* ioOperation Class */

/// <summary>
/// Read radar echo signal into CPU mmemory.
/// </summary>
class ioOperation
{
private:
	std::string m_dirPath;
	std::string m_filePath;
	int m_fileType;  // file polar type

public:
	/// <summary>
	/// Constructing object.
	/// </summary>
	/// <param name="dirPath"></param>
	/// <param name="fileType"></param>
	ioOperation(const std::string& dirPath, const int& fileType);

	~ioOperation();

	/// <summary>
	/// Retrieving basic radar echo signal information.
	/// </summary>
	/// <param name="paras"> radar echo signal parameters </param>
	/// <param name="frameLength"> single frame length </param>
	/// <param name="frameNum"> total frame number</param>
	/// <returns></returns>
	int getSystemParasFirstFileStretch(RadarParameters* paras, int* frameLength, int* frameNum);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="dataN"></param>
	/// <param name="stretchIndex"></param>
	/// <param name="turnAngle"></param>
	/// <param name="pulse_num_all"></param>
	/// <param name="paras"></param>
	/// <param name="frameLength"></param>
	/// <param name="frameNum"></param>
	/// <returns></returns>
	int readKuIFDSALLNBStretch(vec2D_FLOAT* dataN, vec2D_INT* stretchIndex, std::vector<float>* turnAngle, int* pulse_num_all, \
		const RadarParameters& paras, const int& frameLength, const int& frameNum);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="flagDataEnd"></param>
	/// <param name="dataWFileSn"></param>
	/// <param name="dataNOut"></param>
	/// <param name="turnAngleOut"></param>
	/// <param name="dataN"></param>
	/// <param name="paras"></param>
	/// <param name="turnAngle"></param>
	/// <param name="CQ"></param>
	/// <param name="windowHead"></param>
	/// <param name="windowLength"></param>
	/// <param name="nonUniformSampling"></param>
	/// <returns></returns>
	int getKuDatafileSn(int* flagDataEnd, std::vector<int>* dataWFileSn, vec2D_FLOAT* dataNOut, std::vector<float>* turnAngleOut, \
		const vec2D_FLOAT& dataN, const RadarParameters& paras, const std::vector<float>& turnAngle, const int& CQ, const int& windowHead, const int& windowLength, const bool& nonUniformSampling);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="dataW"></param>
	/// <param name="frameHeader"></param>
	/// <param name="stretchIndex"></param>
	/// <param name="dataWFileSn"></param>
	/// <returns></returns>
	int getKuDataStretch(vec1D_COM_FLOAT* dataW, std::vector<int>* frameHeader, \
		const vec2D_INT& stretchIndex, const std::vector<int>& dataWFileSn);

	/// <summary>
	/// write d_data reside in CPU memory back to outFilePath
	/// </summary>
	/// <param name="outFilePath"></param>
	/// <param name="data"></param>
	/// <param name="data_size"></param>
	/// <returns></returns>
	static int writeFile(const std::string& outFilePath, const std::complex<float>* data, const  size_t& data_size);

	/// <summary>
	/// write d_data reside in GPU memory back to outFilePath
	/// </summary>
	/// <param name="outFilePath"></param>
	/// <param name="d_data"></param>
	/// <param name="data_size"></param>
	/// <returns></returns>
	static int dataWriteBack(const std::string& outFilePath, const cuComplex* d_data, const  size_t& data_size);
};

#endif // COMMON_H_
