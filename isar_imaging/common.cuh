#ifndef COMMON_H_
#define COMMON_H_

//#define DATA_WRITE_BACK

// cuda runtime
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <cublas_v2.h>
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


constexpr const char* DIR_PATH = "D:\\Users\\Project\\UnderGraduate_Dissertation\\Data_Code\\_OriginalFile\\Data\\210425235341_047414_1383_00\\";
constexpr auto PI_h = 3.14159265358979f;
constexpr auto lightSpeed_h = 300000000;

namespace fs = std::filesystem;


struct RadarParameters
{
	RadarParameters() = default;
	RadarParameters(int _num_echoes, int _num_range_bins, long long _bandwidth, long long _fc, float _Tp, int _Fs, int _Pos) :
		num_echoes(_num_echoes), num_range_bins(_num_range_bins), band_width(_bandwidth), fc(_fc), Tp(_Tp), Fs(_Fs), Pos(_Pos) {};

	int num_echoes;
	int num_range_bins;
	long long band_width;
	long long fc;
	float Tp;
	int Fs;
	int Pos;
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
void vecMulvec(cublasHandle_t handle, cuComplex* result_matrix, thrust::device_vector<comThr>& vec1, thrust::device_vector<comThr>& vec2, const cuComplex& alpha);


/// <summary>
/// result_matrix = alpah * vec1 * vec2.' + result_matrix;
/// for float number.
/// </summary>
/// <param name="handle"> cublas handle </param>
/// <param name="result_matrix"> multiply result matrix of size m * n (store in column major) </param>
/// <param name="vec1">  vector of length m </param>
/// <param name="vec2"> vecto of length n </param>
/// <param name="alpha"> alpha can be in host or device memory </param>
void vecMulvec(cublasHandle_t handle, float* result_matrix, thrust::device_vector<float>& vec1, thrust::device_vector<float>& vec2, const float& alpha);


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
/// 对距离像进行径向截取
/// </summary>
/// <param name="d_data"> 距离像(格式: 方位*距离, 行主序); </param>
/// <param name="d_data_out"></param>
/// <param name="range_length"> 截取后留下的点数, 一般是2的幂; 输入距离像中心两侧各截取range_length / 2点; </param>
/// <param name="Paras"> 雷达参数 </param>
void cutRangeProfile(cuComplex* d_data, cuComplex* d_data_out, const int range_length, RadarParameters& Paras);


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
/// Generate fftshift array {1, -1, ...}
/// </summary>
/// <param name="fftshift_vec"></param>
void genFFTShiftVec(thrust::device_vector<int>& fftshift_vec);



/// <summary>
/// 获取距离像序列.
/// 对去斜数据, 距离像是对时域回波作fft.
/// </summary>
/// <param name="d_data"> 回波数据 </param>
/// <param name="echo_num"> 慢时间(方位向)点数 </param>
/// <param name="range_num"> 快时间点数 </param>
void getHRRP(cuComplex* d_hrrp, cuComplex* d_data, const int& echo_num, const int& range_num, const thrust::device_vector<int>& fftshift_vec);



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
class ioOperation
{
private:
	std::string m_dirPath;
	std::string m_filePath;
	int m_fileType;  // file polar type

public:
	ioOperation(const std::string& dirPath, const int& fileType);

	~ioOperation();

	int getSystemParasFirstFileStretch(RadarParameters* paras, int* frameLength, int* frameNum);

	int readKuIFDSALLNBStretch(vec2D_FLOAT* dataN, vec2D_INT* stretchIndex, std::vector<float>* turnAngle, int* pulse_num_all, \
		const RadarParameters& paras, const int& frameLength, const int& frameNum);

	int getKuDatafileSn(int* flagDataEnd, std::vector<int>* dataWFileSn, vec2D_FLOAT* dataNOut, std::vector<float>* turnAngleOut, \
		const vec2D_FLOAT& dataN, const RadarParameters& paras, const std::vector<float>& turnAngle, const int& CQ, const int& windowHead, const int& windowLength, const bool& nonUniformSampling);

	int getKuDataStretch(vec1D_COM_FLOAT* dataW, std::vector<int>* frameHeader, \
		const vec2D_INT& stretchIndex, const std::vector<int>& dataWFileSn);

	/// <summary>
	/// write d_data reside in CPU memory back to outFilePath
	/// </summary>
	static int writeFile(const std::string& outFilePath, const std::complex<float>* data, const  size_t& data_size);

	/// <summary>
	/// write d_data reside in GPU memory back to outFilePath
	/// </summary>
	static int dataWriteBack(const std::string& outFilePath, const cuComplex* d_data, const  size_t& data_size);
};

#endif // COMMON_H_
