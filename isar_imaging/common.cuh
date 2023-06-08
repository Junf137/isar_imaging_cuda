#ifndef COMMON_H_
#define COMMON_H_


//#define DATA_WRITE_BACK_DATAW
//#define DATA_WRITE_BACK_HPC
//#define DATA_WRITE_BACK_HRRP
//#define DATA_WRITE_BACK_RA
//#define DATA_WRITE_BACK_PC
//#define DATA_WRITE_BACK_MTRC
//#define DATA_WRITE_BACK_FINAL
//#define SEPARATE_TIMEING_


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
#include <regex>


typedef std::vector<int> vec1D_INT;
typedef std::vector<std::vector<int>> vec2D_INT;
typedef std::vector<float> vec1D_FLT;
typedef std::vector<std::vector<float>> vec2D_FLT;
typedef std::vector<double> vec1D_DBL;
typedef std::vector<std::vector<double>> vec2D_DBL;
typedef std::vector<std::complex<float>> vec1D_COM_FLT;
typedef std::vector<std::vector<std::complex<float>>> vec2D_COM_FLT;
typedef thrust::complex<float> comThr;


constexpr auto PI_FLT = 3.14159265358979f;
constexpr auto PI_DBL = 3.14159265358979;
constexpr auto LIGHT_SPEED = 300000000;
constexpr auto RANGE_NUM_CUT = 512;
constexpr auto RANGE_NUM_IFDS_PC = 4096;
constexpr auto FAST_ENTROPY_ITERATION_NUM = 120;
constexpr auto MAX_THREAD_PER_BLOCK = 1024;
constexpr auto DEFAULT_THREAD_PER_BLOCK = 256;


namespace fs = std::filesystem;


enum DATA_TYPE
{
	DEFAULT = 0,	// 
	IFDS = 1,	// IFDS data
	STRETCH = 2		// stretch data
};

enum POLAR_TYPE
{
	LHP = 0,	// left-hand polarization
	RHP = 1	    // right-hand polarization
};


struct RadarParameters
{
	int echo_num;
	int range_num;
	int data_num;
	long long band_width;
	long long fc;
	double Tp;
	long long Fs;

	// default constructor and equality operator
	RadarParameters() = default;

	bool operator==(const RadarParameters& other) const = default;

	RadarParameters(int echo_num, int range_num, int data_num, long long band_width, long long fc, double Tp, long long Fs)
		: echo_num(echo_num), range_num(range_num), data_num(data_num), band_width(band_width), fc(fc), Tp(Tp), Fs(Fs)
	{
	}
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

	// cuFFT plan after cut range profile
	cufftHandle plan_all_range_c2c;
	cufftHandle plan_all_range_c2c_czt;
	cufftHandle plan_all_echo_c2c_cut;

public:

	/// <summary>
	/// Initializing cuBlas and cuFFT handle
	/// </summary>
	/// <param name="echo_num"></param>
	/// <param name="range_num"></param>
	void handleInit(const int& echo_num, const int& range_num);

	/// <summary>
	/// Destroying cuBlas and cuFFT handle
	/// </summary>
	void handleDest();

	/// <summary>
	/// Default constructor
	/// </summary>
	CUDAHandle() = default;

	/// <summary>
	/// Default Constructor
	/// </summary>
	/// <param name="handle"></param>
	/// <param name="plan_all_echo_c2c"></param>
	/// <param name="plan_all_echo_r2c"></param>
	/// <param name="plan_all_echo_c2r"></param>
	/// <param name="plan_all_range_c2c"></param>
	/// <param name="plan_all_range_c2c_czt"></param>
	/// <param name="plan_all_echo_c2c_cut"></param>
	CUDAHandle(const cublasHandle_t& handle, const cufftHandle& plan_all_echo_c2c, const cufftHandle& plan_all_echo_r2c, const cufftHandle& plan_all_echo_c2r, const cufftHandle& plan_all_range_c2c, const cufftHandle& plan_all_range_c2c_czt, const cufftHandle& plan_all_echo_c2c_cut)
		: handle(handle), plan_all_echo_c2c(plan_all_echo_c2c), plan_all_echo_r2c(plan_all_echo_r2c), plan_all_echo_c2r(plan_all_echo_c2r), plan_all_range_c2c(plan_all_range_c2c), plan_all_range_c2c_czt(plan_all_range_c2c_czt), plan_all_echo_c2c_cut(plan_all_echo_c2c_cut)
	{
	}

	/// <summary>
	/// Equality operator
	/// </summary>
	/// <param name="other"></param>
	/// <returns></returns>
	bool operator==(const CUDAHandle& other) const = default;
};


template <typename T>
/// <summary>
/// 
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void swap_range(T* a, T* b, int len)
{
	// [todo]: separate definition and declaration
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < len) {
		T c = a[tid]; a[tid] = b[tid]; b[tid] = c;
	}
}


template <typename T>
/// <summary>
/// Kernel configuration requirement:
/// (1) block_number == {((range_num / 2) + block.x - 1) / block.x, echo_num}
/// (2) thread_per_block == {256}
/// </summary>
/// <param name="d_data"></param>
/// <param name="cols"></param>
/// <returns></returns>
__global__ void ifftshiftRows(T* d_data, int cols)
{
	// [todo]: separate definition and declaration
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int bidy = blockIdx.y;

	int half_cols = cols >> 1;

	if (tid < half_cols) {
		// swap
		T temp = d_data[bidy * cols + tid];
		d_data[bidy * cols + tid] = d_data[bidy * cols + tid + half_cols];
		d_data[bidy * cols + tid + half_cols] = temp;
	}
}


template <typename T>
/// <summary>
/// Kernel configuration requirement:
/// (1) block_number == {range_num, ((echo_num / 2) + block.x - 1) / block.x}
/// (2) thread_per_block == {256}
/// </summary>
/// <param name="d_data"></param>
/// <param name="cols"></param>
/// <returns></returns>
__global__ void ifftshiftCols(T* d_data, int rows)
{
	// [todo]: separate definition and declaration
	int tid = blockDim.x * blockIdx.y + threadIdx.x;
	int bidx = blockIdx.x;

	int cols = gridDim.x;
	int half_rows = rows >> 1;

	if (tid < half_rows) {
		// swap
		T temp = d_data[tid * cols + bidx];
		d_data[tid * cols + bidx] = d_data[(tid + half_rows) * cols + bidx];
		d_data[(tid + half_rows) * cols + bidx] = temp;
	}
}


/// <summary>
/// 
/// </summary>
/// <param name="handle"></param>
/// <param name="d_vec"></param>
/// <param name="len"></param>
/// <param name="max_idx"></param>
/// <param name="max"></param>
void getMax(cublasHandle_t handle, float* d_vec, int len, int* h_max_idx, float* h_max_val);
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


/// <summary>
/// d_data_conj = conj(d_data);
/// </summary>
/// <param name="d_data"></param>
/// <param name="d_data_conj"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void elementwiseConj(cuComplex* d_data, cuComplex* d_data_conj, int len);


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
__global__ void elementwiseMultiply(cuDoubleComplex* a, cuDoubleComplex* b, cuDoubleComplex* c, int len);
__global__ void elementwiseMultiply(float* a, float* b, float* c, int len);
__global__ void elementwiseMultiply(float* a, cuComplex* b, cuComplex* c, int len);


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
__global__ void elementwiseMultiplyRep(cuDoubleComplex* a, cuDoubleComplex* b, cuDoubleComplex* c, int len_a, int len_b);
__global__ void elementwiseMultiplyRep(float* a, cuComplex* b, cuComplex* c, int len_a, int len_b);


/// <summary>
/// c = b ./ a
/// </summary>
/// <param name="a"></param>
/// <param name="b"></param>
/// <param name="c"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void elementwiseDiv(float* a, cuComplex* b, cuComplex* c, int len);


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
__global__ void diagMulMat(float* d_diag, cuComplex* d_data, cuComplex* d_res, int cols, int len);
__global__ void diagMulMat(double* d_diag, double* d_data, double* d_res, int cols, int len);


/// <summary>
/// res = exp(1j * x)
/// </summary>
/// <param name="x"></param>
/// <param name="res"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void expJ(double* x, cuDoubleComplex* res, int len);


/// <summary>
/// Generating Hamming Window.
/// </summary>
/// <param name="hamming"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void genHammingVec(float* hamming, int len);


//template <typename T>
///// <summary>
///// Reference: https://stackoverflow.com/questions/27925979/thrustmax-element-slow-in-comparison-cublasisamax-more-efficient-implementat
///// </summary>
///// <param name="data"></param>
///// <param name="dsize"></param>
///// <param name="result"></param>
///// <returns></returns>
//__global__ void getMaxIdx(const T* data, const int dsize, int* result);


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
__global__ void sumRows(float* d_data, float* d_sum_rows, int rows, int cols);


/// <summary>
/// Cutting range profile along echo dimension
/// </summary>
/// <param name="d_data_cut"></param>
/// <param name="d_data"></param>
/// <param name="paras"></param>
/// <param name="range_num_cut"></param>
/// <param name="handles"></param>
void cutRangeProfile(cuComplex* d_data_cut, cuComplex* d_data, RadarParameters& paras, const int& range_num_cut, const CUDAHandle& handles);


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
/// setting d_data's element in position of d_index to value val.
/// </summary>
/// <param name="d_data"></param>
/// <param name="d_index"></param>
/// <param name="val"></param>
/// <param name="d_index_len"></param>
/// <returns></returns>
__global__ void setNumInArray(int* d_data, int* d_index, int val, int d_index_len);


/// <summary>
/// d_hrrp = fftshift(fft(d_data))
/// </summary>
/// <param name="d_hrrp"></param>
/// <param name="d_data"></param>
/// <param name="paras"></param>
/// <param name="handles"></param>
void getHRRP(cuComplex* d_hrrp, cuComplex* d_data, const RadarParameters& paras, const CUDAHandle& handles);


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
int turnAngleLine(vec1D_FLT* turnAngle, const vec1D_FLT& azimuth, const vec1D_FLT& pitching);


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
float interpolate(const vec1D_INT& xData, const vec1D_FLT& yData, const int& x, const bool& extrapolate);


/// <summary>
/// 
/// </summary>
/// <param name="dataWFileSn"></param>
/// <param name="dataNOut"></param>
/// <param name="turnAngleOut"></param>
/// <param name="dataN"></param>
/// <param name="turnAngle"></param>
/// <param name="frame_num"></param>
/// <param name="sampling_stride"></param>
/// <param name="window_head"></param>
/// <param name="window_len"></param>
/// <returns></returns>
int uniformSampling(vec1D_INT* dataWFileSn, vec1D_DBL* dataNOut, vec1D_FLT* turnAngleOut, \
	const vec1D_DBL& dataN, const vec1D_FLT& turnAngle, const int& frame_num, const int& sampling_stride, const int& window_head, const int& window_len);


//function [flag_data_end DataW_FileSn DataNOut TurnAngleOut] = NonUniformitySampling(DataN, RadarParameters, TurnAngle, start, M)
int nonUniformSampling();


/* ioOperation Class */
/// <summary>
/// Reading radar signal into CPU memory.
/// </summary>
class ioOperation
{
private:
	std::string m_dir_path;  // directory path
	std::vector<std::string> m_file_vec;
	POLAR_TYPE m_polar_type;  // file polar type
	DATA_TYPE m_data_type;  // file data type

public:
	/// <summary>
	/// Default constructor
	/// </summary>
	ioOperation() = default;

	/// <summary>
	/// Constructor with all filed
	/// </summary>
	/// <param name="m_dir_path"></param>
	/// <param name="m_file_vec"></param>
	/// <param name="m_polar_type"></param>
	/// <param name="m_data_type"></param>
	ioOperation(const std::string& dir_path, const std::vector<std::string>& file_vec, POLAR_TYPE polar_type, DATA_TYPE data_type)
		: m_dir_path(dir_path), m_file_vec(file_vec), m_polar_type(polar_type), m_data_type(data_type)
	{
	}

	/// <summary>
	/// Default destructor
	/// </summary>
	~ioOperation() = default;

	/// <summary>
	/// Default equality operator
	/// </summary>
	/// <param name="other"></param>
	/// <returns></returns>
	bool operator==(const ioOperation& other) const = default;

	/// <summary>
	/// Object initialization(IFDS and STRETCH mode).
	/// </summary>
	/// <param name="INTERMEDIATE_DIR"></param>
	/// <param name="dir_path"></param>
	/// <param name="polar_type"></param>
	/// <param name="data_type"></param>
	void ioInit(std::string* INTERMEDIATE_DIR, const std::string& dir_path, const POLAR_TYPE& polar_type, const DATA_TYPE& data_type);

	/// <summary>
	/// Get system parameters from the first file(IFDS and STRETCH mode).
	/// </summary>
	/// <param name="paras"></param>
	/// <param name="frame_len"></param>
	/// <param name="frame_num"></param>
	/// <returns></returns>
	int getSystemParas(RadarParameters* paras, int* frame_len, int* frame_num);

	/// <summary>
	/// Read narrow-band information from all matched file under given directory(IFDS and STRETCH mode).
	/// [todo] update data parsing protocol in STRETCH mode based on new parsing file('Read_Ku_IFDS_ALLNB_Stretch.m').
	/// </summary>
	/// <param name="dataN"></param>
	/// <param name="turnAngle"></param>
	/// <param name="paras"></param>
	/// <param name="frame_len"></param>
	/// <param name="frame_num"></param>
	/// <returns></returns>
	int readKuIFDSAllNB(vec1D_DBL* dataN, vec1D_FLT* turnAngle, \
		const RadarParameters& paras, const int& frame_len, const int& frame_num);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="dataW"></param>
	/// <param name="frame_len"></param>
	/// <param name="dataWFileSn"></param>
	/// <param name="window_len"></param>
	/// <returns></returns>
	int getKuData(vec1D_COM_FLT* dataW, \
		const int& frame_len, const vec1D_INT& dataWFileSn, const int& window_len);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="dataW"></param>
	/// <param name="frame_len"></param>
	/// <param name="dataWFileSn"></param>
	/// <param name="window_len"></param>
	/// <returns></returns>
	int getKuDataStretch(vec1D_COM_FLT* dataW, \
		const int& frame_len, const vec1D_INT& dataWFileSn, const int& window_len);

	/// <summary>
	/// write data reside in CPU memory back to outFilePath
	/// </summary>
	/// <param name="outFilePath"></param>
	/// <param name="data"></param>
	/// <param name="data_size"></param>
	/// <returns></returns>
	static int writeFile(const std::string& outFilePath, const cuComplex* data, const  size_t& data_size);
	static int writeFile(const std::string& outFilePath, const cuDoubleComplex* data, const  size_t& data_size);
	static int writeFile(const std::string& outFilePath, const float* data, const  size_t& data_size);
	static int writeFile(const std::string& outFilePath, const double* data, const  size_t& data_size);

	/// <summary>
	/// write d_data reside in GPU memory back to outFilePath
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="outFilePath"></param>
	/// <param name="d_data"></param>
	/// <param name="data_size"></param>
	/// <returns></returns>
	template <typename T>
	static int dataWriteBack(const std::string& outFilePath, const T* d_data, const  size_t& data_size)
	{
		T* h_data = new T[data_size];

		checkCudaErrors(cudaMemcpy(h_data, d_data, sizeof(T) * data_size, cudaMemcpyDeviceToHost));

		ioOperation::writeFile(outFilePath, h_data, data_size);

		delete[] h_data;
		h_data = nullptr;

		return EXIT_SUCCESS;
	}
};

#endif // COMMON_H_
