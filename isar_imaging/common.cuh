#ifndef COMMON_H_
#define COMMON_H_


//#define DATA_WRITE_BACK_DATAW
//#define DATA_WRITE_BACK_HPC
//#define DATA_WRITE_BACK_HRRP
#define DATA_WRITE_BACK_RA
#define DATA_WRITE_BACK_FINAL


#define MIN(a, b) (a<b) ? a : b
#define MAX(a, b) (a>b) ? a : b


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
constexpr auto RANGE_NUM_CUT = 512;
constexpr auto FAST_ENTROPY_ITERATION_NUM = 120;

namespace fs = std::filesystem;


struct RadarParameters
{
	int echo_num;
	int range_num;
	int data_num;
	long long band_width;
	long long fc;
	float Tp;
	int Fs;
};


class CUDAHandle {
public:
	// * Overall cuBlas handle
	cublasHandle_t handle;

	// * Overall cuFFT plan
	cufftHandle plan_all_echo_c2c;
	//cufftHandle plan_one_echo_c2c;  // used in range alignment iteration version
	//cufftHandle plan_one_echo_r2c;  // implicitly forward
	cufftHandle plan_all_echo_r2c;  // implicitly forward
	//cufftHandle plan_one_echo_c2r;  // implicitly inverse
	cufftHandle plan_all_echo_c2r;  // implicitly inverse

	cufftHandle plan_all_range_c2c;

public:

	CUDAHandle(const int& echo_num, const int& range_num);

	~CUDAHandle();

};


/// <summary>
/// result_matrix = alpha * vec1 * vec2.' + result_matrix;
/// for cuComplex number.
/// </summary>
/// <param name="handle"> cublas handle </param>
/// <param name="result_matrix"> multiply result matrix of size m * n (store in column major) </param>
/// <param name="vec1">  vector of length m </param>
/// <param name="vec2"> vector of length n </param>
/// <param name="alpha"> alpha can be in host or device memory </param>
void vecMulvec(cublasHandle_t handle, cuComplex* result_matrix, thrust::device_vector<comThr>& vec1, thrust::device_vector<comThr>& vec2, const cuComplex& alpha);  // todo: deprecated
void vecMulvec(cublasHandle_t handle, cuComplex* d_vec1, int len1, cuComplex* d_vec2, int len2, cuComplex* d_res_matrix, const cuComplex& alpha);


/// <summary>
/// result_matrix = alpha * vec1 * vec2.' + result_matrix;
/// for float number.
/// </summary>
/// <param name="handle"> cublas handle </param>
/// <param name="result_matrix"> multiply result matrix of size m * n (store in column major) </param>
/// <param name="vec1">  vector of length m </param>
/// <param name="vec2"> vector of length n </param>
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


/// <summary>
/// 
/// </summary>
/// <param name="a"></param>
/// <param name="abs"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void elementwiseAbs(cuComplex* a, float* abs, int len);


//__global__ void elementwiseMean(cuComplex* a, cuComplex* b, cuComplex* c, int len);


/// <summary>
/// Perform element-wise vector multiplication.
/// c = a .* b
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="c"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void elementwiseMultiply(cuComplex* a, cuComplex* b, cuComplex* c, int len);


/// <summary>
/// Perform element-wise vector multiplication.
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
/// Perform element-wise vector multiplication.
/// c = a .* b
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="c"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void elementwiseMultiply(float* a, float* b, float* c, int len);


/// <summary>
/// c = b ./ repmat(a, (len_b/len_a), 1)
/// </summary>
/// <param name="a"> len_a </param>
/// <param name="b"> len_b </param>
/// <param name="c"> len_c == len_b </param>
/// <param name="len_a"></param>
/// <param name="len_b"></param>
/// <returns></returns>
__global__ void elementwiseDivRep(float* a, float* b, float* c, int len_a, int len_b);


/// <summary>
/// Perform element-wise vector multiplication.
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
/// Perform element-wise vector multiplication.
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
///// Reference: https://stackoverflow.com/questions/27925979/thrustmax-element-slow-in-comparison-cublasisamax-more-efficient-implementat
///// </summary>
///// <param name="data"></param>
///// <param name="dsize"></param>
///// <param name="result"></param>
///// <returns></returns>
//__global__ void getMaxIdx(const T* data, const int dsize, int* result);


///// <summary>
///// get maximum value of every column
///// Reference:  https://stackoverflow.com/questions/17698969/determining-the-least-element-and-its-position-in-each-matrix-column-with-cuda-t
///// </summary>
///// <param name="c"> data set, column is not coalescing </param>
///// <param name="maxval"> vector to store maximum value </param>
///// <param name="maxidx"> vector to store maximum value's index </param>
///// <param name="row"> number of rows </param>
///// <param name="col"> number of columns ></param>
//void getMaxInColumns(thrust::device_vector<float>& c, thrust::device_vector<float>& maxval, thrust::device_vector<int>& maxidx, int row, int col);


/// <summary>
/// Getting the max element of every single column in matrix d_data.
/// Each block is responsible for the calculation of a single column.
/// Since d_data is arranged by row-major in device memory and window length(echo_num) used in ISAR imaging is generally bounded inside {256, 512}.
/// Therefore element per column won't be so much bigger than number of thread per block like the situation in sumRows.
/// In this case, we can discard the partial maximum calculation in each thread and directly perform reduction of shared memory in block-scale.
/// Kernel configuration requirements:
/// (1) block_number == cols
/// (2) shared_memory_number == thread_per_block == rows
/// </summary>
/// <param name="d_data"></param>
/// <param name="d_max_clos"></param>
/// <param name="rows"></param>
/// <param name="cols"></param>
/// <returns></returns>
__global__ void maxCols(float* d_data, float* d_max_clos, int rows, int cols);


///// <summary>
///// Expanding index.
///// Expanding the number of every element in vector starting from first2 to the corresponding value in vector starting from first1 and ending at end1.
///// first {2,2,2}. second {1,2,3}. output {1,1,2,2,3,3}.
///// </summary>
///// <typeparam name="InputIterator1"></typeparam>
///// <typeparam name="InputIterator2"></typeparam>
///// <typeparam name="OutputIterator"></typeparam>
///// <param name="first1"></param>
///// <param name="last1"></param>
///// <param name="first2"></param>
///// <param name="output"></param>
///// <returns></returns>
//template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
//OutputIterator expand(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator output);
//
//
///// <summary>
///// Calculating summation of each row of matrix d_data alone the second dimension.
///// </summary>
///// <param name="d_data"></param>
///// <param name="d_sum_rows"></param>
///// <param name="row"></param>
///// <param name="col"></param>
//void sumRows_thr(cuComplex* d_data, cuComplex* d_sum_rows, const int& row, const int& col);


/// <summary>
/// Calculate summation of every column. Equal to sum(d_data, 1) in matlab.
/// Each block is responsible for summing a single column.
/// Since d_data is arranged by row-major in device memory and window length(echo_num) used in ISAR imaging is generally bounded inside {256, 512}.
/// Therefore element per column won't be so much bigger than number of thread per block like the situation in sumRows.
/// In this case, we can discard the partial sum calculation in each thread and directly perform reduction of shared memory in block-scale.
/// Kernel configuration requirements:
/// (1) block_number == cols
/// (2) shared_memory_number == thread_per_block == rows
/// </summary>
/// <param name="d_data"></param>
/// <param name="rows"></param>
/// <param name="cols"></param>
/// <param name="sum_clos"></param>
/// <returns></returns>
__global__ void sumCols(float* d_data, float* sum_clos, int rows, int cols);


/// <summary>
/// Calculate summation of every rows. Equal to sum(d_data, 2) in matlab.
/// Each block is responsible for summing a single row.
/// Since d_data is arranged by row-major in device memory and element per row is usually much bigger than number of thread per block.
/// Therefore calculating partial sum in each thread before performing reduction on shared memory in block-scale is necessary.
/// Kernel configuration requirements:
/// (1) block_number == rows
/// (2) shared_memory_number == thread_per_block == {256, 512, 1024}
/// </summary>
/// <param name="d_data"></param>
/// <param name="rows"></param>
/// <param name="cols"></param>
/// <param name="sum_rows"></param>
/// <returns></returns>
__global__ void sumRows(cuComplex* d_data, cuComplex* sum_rows, int rows, int cols);


/// <summary>
/// Cutting range profile along echo dimension
/// </summary>
/// <param name="d_data"></param>
/// <param name="paras"></param>
/// <param name="range_num_cut"></param>
/// <param name="handle"></param>
void cutRangeProfile(cuComplex*& d_data, RadarParameters& paras, const int& range_num_cut, const CUDAHandle& handles);


/// <summary>
/// Cutting range profile in range dimension along with echo dimension
/// </summary>
/// <param name="d_in"> d_data before interception </param>
/// <param name="d_out"> d_data after interception </param>
/// <param name="data_size"> size of d_data after interception  </param>
/// <param name="offset"> starting point of each echo interception </param>
/// <param name="num_elements"> point number of every echo after interception </param>
/// <param name="num_ori_elements"> point number of every echo before interception </param>
/// <returns></returns>
__global__ void cutRangeProfileHelper(cuComplex* d_in, cuComplex* d_out, const int data_size,
	const int offset, const int num_elements, const int num_ori_elements);


/// <summary>
/// Calculating 2^nextpow2(N).
/// power of 2 closest to N.
/// </summary>
/// <param name="N"></param>
/// <returns></returns>
int nextPow2(int N);


/// <summary>
/// 将array中位于index处的元素置为set_num
/// </summary>
/// <param name="arrays"> data array </param>
/// <param name="index"> index of element to be replaces </param>
/// <param name="set_num"> replace value </param>
/// <param name="num_index"> number of element to be replaces </param>
/// <returns></returns>

/// <summary>
/// setting d_data's element in position of d_index to value val.
/// </summary>
/// <param name="d_data"></param>
/// <param name="d_index"></param>
/// <param name="val"></param>
/// <param name="d_index_len"></param>
/// <returns></returns>
__global__ void setNumInArray(int* d_data, int* d_index, int val, int d_index_len);


/// <summary>
/// HRRP
/// </summary>
/// <param name="plan"></param>
/// <param name="d_hrrp"></param>
/// <param name="d_data"></param>
/// <param name="echo_num"></param>
/// <param name="range_num"></param>
/// <param name="d_fftshift"></param>
void getHRRP(cuComplex* d_hrrp, cuComplex* d_data, float* hamming, const RadarParameters& paras, const CUDAHandle& handles);


/// <summary>
/// Calculate target rotation angle, calculate unit vector of two radar's observation direction
/// </summary>
/// <param name="azimuth1"></param>
/// <param name="pitching1"></param>
/// <param name="azimuth2"></param>
/// <param name="pitching2"></param>
/// <returns></returns>
float getTurnAngle(const float& azimuth1, const float& pitching1, const float& azimuth2, const float& pitching2);


/// <summary>
/// Calculating target rotation angle curve.
/// finding lost point, then estimate turn angle for each segment, finally add them together.
/// </summary>
/// <param name="turnAngle"> target rotation angle curve </param>
/// <param name="azimuth">  </param>
/// <param name="pitching">  </param>
/// <returns></returns>
int turnAngleLine(std::vector<float>* turnAngle, const std::vector<float>& azimuth, const std::vector<float>& pitching);


/// <summary>
/// Returns interpolated value at x from parallel arrays ( xData, yData )
/// Assumes that xData has at least two elements, is sorted and is strictly monotonic increasing
/// boolean argument extrapolate determines behavior beyond ends of array (if needed)
/// </summary>
/// <param name="xData"></param>
/// <param name="yData"></param>
/// <param name="x"> interpolation point </param>
/// <param name="extrapolate"> option for extrapolate </param>
/// <returns></returns>
float interpolate(const std::vector<int>& xData, const std::vector<float>& yData, const int& x, const bool& extrapolate);


// [flag_data_end DataW_FileSn DataNOut TurnAngleOut] = UniformitySampling(DataN, TurnAngle, sampling_stride, WindowHead, WindowLength)
int uniformSamplingFun(int* flagDataEnd, std::vector<int>* dataWFileSn, vec2D_FLOAT* dataNOut, std::vector<float>* turnAngleOut, \
	const vec2D_FLOAT& dataN, const std::vector<float>& turnAngle, const int& sampling_stride, const int& window_head, const int& window_len);


//function [flag_data_end DataW_FileSn DataNOut TurnAngleOut] = NonUniformitySampling(DataN, RadarParameters, TurnAngle, start, M)
int nonUniformSamplingFun();


/* ioOperation Class */

/// <summary>
/// Read radar echo signal into CPU memory.
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
	/// <param name="frame_len"> single frame length </param>
	/// <param name="frame_num"> total frame number</param>
	/// <returns></returns>
	int getSystemParasFirstFileStretch(RadarParameters* paras, int* frame_len, int* frame_num);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="dataN"></param>
	/// <param name="stretchIndex"></param>
	/// <param name="turnAngle"></param>
	/// <param name="pulse_num_all"></param>
	/// <param name="paras"></param>
	/// <param name="frame_len"></param>
	/// <param name="frame_num"></param>
	/// <returns></returns>
	int readKuIFDSALLNBStretch(vec2D_FLOAT* dataN, vec2D_INT* stretchIndex, std::vector<float>* turnAngle, int* pulse_num_all, \
		const RadarParameters& paras, const int& frame_len, const int& frame_num);

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
	/// <param name="sampling_stride"></param>
	/// <param name="window_head"></param>
	/// <param name="window_len"></param>
	/// <param name="nonUniformSampling"></param>
	/// <returns></returns>
	int getKuDatafileSn(int* flagDataEnd, std::vector<int>* dataWFileSn, vec2D_FLOAT* dataNOut, std::vector<float>* turnAngleOut, \
		const vec2D_FLOAT& dataN, const RadarParameters& paras, const std::vector<float>& turnAngle, const int& sampling_stride, const int& window_head, const int& window_len, const bool& nonUniformSampling);

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
