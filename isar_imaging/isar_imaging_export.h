#ifndef ISAR_IMAGING_EXPORT_H_
#define ISAR_IMAGING_EXPORT_H_

#include <string>
#include <vector>
#include <complex>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>


#define DLL_EXPORT_API __declspec(dllexport)  // [todo] dllexport ? dllimport


/**********************
 * This header file exports main imaging function as c++ API using c-style naming standard
 **********************/


 // Duplicate from 'common.cuh'
typedef std::vector<int> vec1D_INT;
typedef std::vector<std::vector<int>> vec2D_INT;
typedef std::vector<float> vec1D_FLT;
typedef std::vector<std::vector<float>> vec2D_FLT;
typedef std::vector<double> vec1D_DBL;
typedef std::vector<std::vector<double>> vec2D_DBL;
typedef std::vector<std::complex<float>> vec1D_COM_FLT;
typedef std::vector<std::vector<std::complex<float>>> vec2D_COM_FLT;


/// <summary>
/// Init GPU Device.
/// Pick the device with highest Gflops/s. (single GPU mode)
/// </summary>
/// <returns></returns>
extern "C" DLL_EXPORT_API int gpuDevInit();


/// <summary>
/// Pre-processing data.
/// This function should be called only once for any file.
/// </summary>
/// <param name="dataN"></param>
/// <param name="turnAngle"></param>
/// <param name="frame_len"></param>
/// <param name="frame_num"></param>
/// <param name="dir_path"></param>
/// <param name="polar_type"></param>
/// <param name="data_type"></param>
/// <returns></returns>
extern "C" DLL_EXPORT_API int dataParsing(vec1D_DBL * dataN, vec1D_FLT * turnAngle, int* frame_len, int* frame_num, \
	const std::string & dir_path, const int& polar_type, const int& data_type);


/// <summary>
/// Extracting data from file for every single imaging process.
/// </summary>
/// <param name="dataWFileSn"></param>
/// <param name="dataNOut"></param>
/// <param name="turnAngleOut"></param>
/// <param name="dataW"></param>
/// <param name="dataN"></param>
/// <param name="turnAngle"></param>
/// <param name="frame_len"></param>
/// <param name="frame_num"></param>
/// <param name="sampling_stride"></param>
/// <param name="window_head"></param>
/// <param name="window_len"></param>
/// <param name="data_type"></param>
/// <returns></returns>
extern "C" DLL_EXPORT_API int dataExtracting(vec1D_INT * dataWFileSn, vec1D_DBL * dataNOut, vec1D_FLT * turnAngleOut, vec1D_COM_FLT * dataW, \
	const vec1D_DBL & dataN, const vec1D_FLT & turnAngle, const int& frame_len, const int& frame_num, const int& sampling_stride, const int& window_head, const int& window_len, const int& data_type);


/// <summary>
/// Initializing CPU and GPU memory.
/// This function should be called only once for any file.
/// </summary>
/// <param name="h_img"></param>
/// <param name="dataWFileSn"></param>
/// <param name="dataNOut"></param>
/// <param name="turnAngleOut"></param>
/// <param name="dataW"></param>
/// <param name="window_len"></param>
/// <param name="frame_len"></param>
/// <param name="data_type"></param>
/// <returns></returns>
extern "C" DLL_EXPORT_API void imagingMemInit(vec1D_FLT * h_img, vec1D_INT * dataWFileSn, vec1D_DBL * dataNOut, vec1D_FLT * turnAngleOut, vec1D_COM_FLT * dataW, \
	const int& window_len, const int& frame_len, const int& data_type);


/// <summary>
/// ISAR imaging process for single image.
/// Result is a matrix of size (echo_num * range_num), stored in memory pointed by h_img.
/// </summary>
/// <param name="h_img"></param>
/// <param name="data_type"></param>
/// <param name="h_data"></param>
/// <param name="dataNOut"></param>
/// <param name="option_alignment"></param>
/// <param name="option_phase"></param>
/// <param name="if_hpc"></param>
/// <param name="if_mtrc"></param>
/// <returns></returns>
extern "C" DLL_EXPORT_API void isarMainSingle(float* h_img, \
	const int& data_type, const std::complex<float>*h_data, const vec1D_DBL & dataNOut, const int& option_alignment, const int& option_phase, const bool& if_hpc, const bool& if_mtrc);


/// <summary>
/// 
/// </summary>
/// <param name="outFilePath"></param>
/// <param name="data"></param>
/// <param name="data_size"></param>
/// <returns></returns>
extern "C" DLL_EXPORT_API void writeFileFLT(const std::string& outFilePath, const float* data, const  size_t& data_size);


/// <summary>
/// Free allocated memory in CPU and GPU.
/// Destroy pointer.
/// </summary>
/// <returns></returns>
extern "C" DLL_EXPORT_API void imagingMemDest();


/******************
 * API for Simulation Data
 ******************/

/// <summary>
/// Initializing parameters for ISAR imaging.
/// </summary>
/// <param name="h_img"> nullptr required </param>
/// <param name="echo_num"> echo number in slow time </param>
/// <param name="range_num"> range number in fast time </param>
/// <param name="band_width"> </param>
/// <param name="fc"> </param>
/// <param name="Fs"> </param>
/// <param name="Tp"> </param>
//extern "C" DLL_EXPORT_API void parasInit(float** h_img, \
//	const int& echo_num, const int& range_num, const long long& band_width, const long long& fc, const int& Fs, const double& Tp);


/// <summary>
/// 
/// </summary>
/// <param name="h_data"></param>
/// <param name="dataNOut"></param>
/// <param name="info_mat"></param>
/// <param name="real_mat"></param>
/// <param name="imag_mat"></param>
//void sim_data_extract(std::complex<float>* h_data, std::vector<std::vector<double>>* dataNOut, \
//	const char* info_mat, const char* real_mat, const char* imag_mat);

#endif // !ISAR_IMAGING_EXPORT_H_

