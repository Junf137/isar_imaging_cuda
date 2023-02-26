// 一些公用的模块

#ifndef COMMON_H_
#define COMMON_H_
#define _USE_MATH_DEFINES

#include <cstdlib>
#include <complex>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"
#include "cufft.h"
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
#include <thrust/extrema.h>
#include <thrust/iterator/discard_iterator.h>



/*************************************
 * 以下为常量和错误检查的预定义部分
 *************************************/
typedef thrust::complex<float> comThr;
//#define PI 3.14159265358979323846
__constant__ float PI = 3.14159265358979f;
__constant__ float lightSpeed = 300000000;
const float PI_h = 3.14159265358979f;
const float lightSpeed_h = 300000000;
//#define PI 3.141592653589793
//#define lightSpeed 300000000

#define checkCudaErrors(a) do{\
	if(cudaSuccess != (a)) {\
		fprintf(stderr, "Cuda runtime error in line %d of file %s \
				: %s\n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError())); \
        system("pause");\
	    exit(EXIT_FAILURE); \
	} \
}while(0);

#define checkCublasErrors(a) do{\
if(CUBLAS_STATUS_SUCCESS != (a)) {\
		fprintf(stderr, "Cuda BLAS error in line %d of file %s",__LINE__, __FILE__);\
	    system("pause");\
	    exit(EXIT_FAILURE);\
	}\
}while(0);

#define checkCuFFTErrors(a) do{\
if(CUFFT_SUCCESS != (a)) {\
		fprintf(stderr, "Cuda FFT error in line %d of file %s ",__LINE__, __FILE__);\
        system("pause");\
	    exit(EXIT_FAILURE);\
	}\
}while(0);



/**************************************
 * 以下为雷达参数和网络的相关结构体
 **************************************/
struct RadarParameters
{
	RadarParameters() = default;
	RadarParameters(unsigned int _num_echoes, unsigned int _num_range_bins, long long _bandwidth, long long _fc, float _Tp, unsigned int _Fs, int _Pos) :\
		num_echoes(_num_echoes), num_range_bins(_num_range_bins), band_width(_bandwidth), fc(_fc), Tp(_Tp), Fs(_Fs), Pos(_Pos) {};

	unsigned int num_echoes;
	unsigned int num_range_bins;
	long long band_width;
	long long fc;
	float Tp;
	unsigned int Fs;
	int Pos;
	int cLen;
	double kai;
};

typedef struct _PACKAGE_HEAD
{
	unsigned int PacketMark;//0x55AAAA55
	unsigned int PacketID;//每个ID对应一种数据类型，如果此父包含多种类型的子包，此变量可不填，直接去对应填每个子包头MSGID即可。
	unsigned int PacketCnt;//父包帧计数，多用于以一定的频次(如一秒发一次)且需要验证相邻包不重复性或者递增性的传输时用，如只是控制指令或者不需要辨别相邻父包之间的不重复性或者递增性时，此变量可不填
	unsigned int PacketLen;//父包除去父包头的所有数据的长度，数据长度=子包头的长度+子包数据内容的长度
	unsigned int MSGNum;//父包中子包的数量

	unsigned int Year;//年 基于北京时间 2019
	unsigned int Month;//月 5
	unsigned int Day;//日 19
	unsigned int Hour;//时 11
	unsigned int Miniute;//分 0
	unsigned int Second;//秒 12
	unsigned int mSecond;//秒内毫秒 12
	unsigned int uSecond;//毫秒内微秒 12
	unsigned int Version; //软件版本号
	unsigned int Reserved[2];//预留

}PACKAGE_HEAD;

typedef struct _MSG_HEAD
{
	unsigned int MSGMark;//0xAA5555AA

	unsigned int MSGLen;//子包数据内容的长度

	//子包数据内容相对于源数据偏移
	//辅助接收端用于拼接数据时确定此子包数据内容在源数据的的相对位置
	unsigned int MSGOffset;

	//如果父包中包含多种类型的子包，那么MSGID按各自所属的数据类型的ID填写
	//如果父包中只含一种类型数据的子包，那么MSGID==PacketID
	unsigned int MSGID; //子包ID 
	//如果父包还是以一定的频次发送的，那就子包头的PacketCnt要填写和父包头一样
	//这样如果一种源数据以一定频次(如一秒一次)，且每次数据都超过了网络单次传输最大包长	//度的限制(如64KB)需要分N个父包传，有了ID和Cnt的配合，就可以确认当下收到的这一父包	//中的子包属于某一时刻的某源数据
	unsigned int PacketCnt; //父包帧计数

	unsigned int Reserved[2]; //预留

}MSG_HEAD;




/********************************
* 以下为thrust库所用functor
* 备注：之后了解到使用lambda而不是结构体更简洁，
*       但涉及到大量的代码修改，于是依旧使用结构来实现functor
********************************/
// ======== 求平方 ======== //
template<typename T>
struct square_functor
{
	typedef T first_argument_type;
	typedef T second_argument_type;
	typedef T result_type;

	__host__ __device__
		float operator()(const T& x) const {
		return x * x;
	}
};

// ======== 两个复数相乘 ======== //
struct Complex_Mul_Complex :public thrust::binary_function<comThr, comThr, comThr>
{
	__host__ __device__
		comThr operator() (comThr a, comThr b) const {
		return a * b;
	}
};

// ======== 生成hamming窗 ======== //
struct Hamming_Window : public thrust::unary_function<float, float>
{
	const int NumEcho;
	Hamming_Window(int _NumEcho) :NumEcho(_NumEcho) {}
	__host__ __device__
		float operator()(float x)
	{
		return (0.54f - 0.46f * cos(2 * PI * (x / NumEcho - 1)));
	}
};

// ======== IFFT数据归一化后乘以另一复数 ======== //
// (a / N) * b
struct ComplexIfft_Mul_Complex :public thrust::binary_function<comThr, comThr, comThr>
{
	const int nfft;
	ComplexIfft_Mul_Complex(int _nfft) : nfft(_nfft) {};
	__host__ __device__
		comThr operator() (comThr a, comThr b) const {
		return (a / float(nfft)) * b;
	}
};

// ======== IFFT数据除以N ======== //
struct ifftNor :public thrust::unary_function<comThr, comThr>
{
	const int nfft;
	ifftNor(int _nfft) : nfft(_nfft) {};
	__host__ __device__
		comThr operator() (comThr a) const {
		return (a / float(nfft));
	}
};

// ======== 求向量的绝对值 ======== //
struct Get_abs :public thrust::unary_function<comThr, float>
{
	__host__ __device__
		float operator() (comThr a) const {
		return thrust::abs(a);
	}
};

// ======== 共轭相乘 ======== //
struct conjComplex_Mul_Complex : public thrust::binary_function<comThr, comThr, comThr>
{
	__host__ __device__
		comThr operator()(comThr x, comThr y)
	{
		return thrust::conj(x) * y;
	}
};

// ======== 一般用于向量最大值归一化 ======== //
template<typename T>
struct maxNormalize :public thrust::unary_function<T, T>
{
	const T normalize_factor;
	maxNormalize(T _normalize_factor) : normalize_factor(_normalize_factor) {};
	__host__ __device__
		float operator()(T& x) const {
		return float(x) / float(normalize_factor);
	}
};

// ======== 向量乘以一常数 ======== //
template<typename T>
struct vectorAmplify :public thrust::unary_function<T, T>
{
	const T amplify_factor;
	vectorAmplify(T _amplify_factor) : amplify_factor(_amplify_factor) {};
	__host__ __device__
		T operator()(T& x) const {
		return amplify_factot * x;
	}
};

struct Hanning_Window : public thrust::unary_function<float, float>
{
	const int NumEcho;
	Hanning_Window(int _NumEcho) :NumEcho(_NumEcho) {}
	__host__ __device__
		float operator()(float x)
	{
		return (0.54f - 0.46f * cos(2.0 * PI * (x / NumEcho - 1)));
	}
};

struct Float_Mul_Complex :public thrust::binary_function<float, comThr, comThr>
{
	__host__ __device__
		comThr operator() (float a, comThr b) const {
		return a * b;
	}
};

/*********************
 * 通用函数
 *********************/

 // 打印设备（GPU）信息
void GetDeviceInformation(int device_num);

// 对输入数据求距离像
void GetHRRP(cuComplex* d_data, unsigned int echo_num, unsigned int range_num, const std::string& arrangement_rank);

// 利用thrust库进行fft
void fftshiftThrust(cuComplex* d_data, int data_length);

void fftshiftMulWay(thrust::device_vector<int>& shift_vec, size_t length);

// 两个向量相乘（复数）
void vectorMulvectorCublasC(cublasHandle_t handle, cuComplex* result_matrix, thrust::device_vector<comThr>& vec1, thrust::device_vector<comThr>& vec2, int m, int n);

// 两个向量相乘（实数）
void vectorMulvectorCublasf(cublasHandle_t handle, float* result_matrix, thrust::device_vector<float>& vec1, thrust::device_vector<float>& vec2, int m, int n);

// 求2的幂
int nextPow2(int N);

// 裁剪距离像
void cutRangeProfile(cuComplex* d_data, cuComplex* d_data_out, const int range_length, RadarParameters& Paras);

// 求每列最大值
void getMaxInColumns(thrust::device_vector<float>& c, thrust::device_vector<float>& maxval, thrust::device_vector<int>& maxidx, int row, int col);

// 加hamming窗
void Add_HanningWindow(cuComplex* d_DataCut, int NumEcho, int NumRange);

// 将array中位于index处的元素置为set_num
template <typename T>
__global__ void setNumInArray(T* arrays, int* index, T set_num, int num_index);

/**************************************************
* 函数功能：将array中位于index处的元素置为set_num
* 输入参数：
* array:    数据序列；
* index:    要被替换的元素索引
* set_num： 替换值
* num_index:元素个数
**************************************************/
template <typename T>
__global__ void setNumInArray(T* arrays, int* index, T set_num, int num_index)
{
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= num_index)
		return;
	arrays[index[tid]] = set_num;
}

// 数据按回波存放时，距离像径向截取的辅助函数
__global__ void cutRangeProfileHelper(cuComplex* d_in, cuComplex* d_out, const int data_size,
	const int offset, const int num_elements, const int num_ori_elements);

/**********************************
 * 读写文件类
 **********************************/
class ioOperation {
public:

	int ReadFile(std::string& path_real, std::string& path_imag, std::complex<float>* data);
	//将String按照Hex的形式输出
	std::string string_to_hex(const std::string& in);

	//Find the narrowband parameters from the head of the data
	int FindPara(std::string& path, RadarParameters& para);

	//Put data in the matrix
	int ReadData(std::string& path, std::complex<float>* data, float* vel, float* range, float* pit, float* azu, RadarParameters& para);

	int ReadFile(std::string& path, float* data);

	int WriteFile(std::string& path_real, std::string& path_imag, std::complex<float>* data, size_t data_size);

	int WriteFile(std::string& path, float* data, size_t data_size);

	int WriteFile(std::string& path, std::complex<float>* data, size_t data_size);
};

#endif /* COMMON_H_ */