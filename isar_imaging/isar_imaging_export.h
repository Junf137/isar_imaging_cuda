#ifndef ISAR_IMAGING_EXPORT_H_
#define ISAR_IMAGING_EXPORT_H_

#include <string>
#include <vector>
#include <complex>


#define DLL_EXPORT_API __declspec(dllexport)  // [todo] dllexport ? dllimport


/// ---* This header file export main imaging function in c++ API using c-style naming standard *---
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
/// <param name="stretchIndex"></param>
/// <param name="turnAngle"></param>
/// <param name="frame_len"></param>
/// <param name="frame_num"></param>
/// <param name="dir_path"></param>
/// <returns></returns>
extern "C" DLL_EXPORT_API int dataParsing(vec2D_DBL * dataN, vec1D_INT * stretchIndex, vec1D_FLT * turnAngle, int* frame_len, int* frame_num, \
	const std::string & dir_path);


/// <summary>
/// Extracting data from file for every single imaging process.
/// </summary>
/// <param name="dataWFileSn"></param>
/// <param name="dataNOut"></param>
/// <param name="turnAngleOut"></param>
/// <param name="dataW"></param>
/// <param name="dataN"></param>
/// <param name="stretchIndex"></param>
/// <param name="frame_len"></param>
/// <param name="turnAngle"></param>
/// <param name="sampling_stride"></param>
/// <param name="window_head"></param>
/// <param name="window_len"></param>
/// <returns></returns>
extern "C" DLL_EXPORT_API int dataExtracting(vec1D_INT * dataWFileSn, vec2D_DBL * dataNOut, vec1D_FLT * turnAngleOut, vec1D_COM_FLT * dataW, \
	const vec2D_DBL & dataN, const vec1D_INT & stretchIndex, const int frame_len, const vec1D_FLT & turnAngle, const int& sampling_stride, const int& window_head, const int& window_len);


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
/// <returns></returns>
extern "C" DLL_EXPORT_API void imagingMemInit(float** h_img, vec1D_INT * dataWFileSn, vec2D_DBL * dataNOut, vec1D_FLT * turnAngleOut, vec1D_COM_FLT * dataW, \
	const int& window_len, const int& frame_len);


/// <summary>
/// ISAR imaging process for single image.
/// Result is a matrix of size (echo_num * range_num), stored in memory pointed by h_img.
/// </summary>
/// <param name="h_img"></param>
/// <param name="data_style"></param>
/// <param name="h_data"></param>
/// <param name="dataNOut"></param>
/// <param name="option_alignment"></param>
/// <param name="option_phase"></param>
/// <param name="if_hpc"></param>
/// <param name="if_mtrc"></param>
/// <returns></returns>
extern "C" DLL_EXPORT_API void isarMainSingle(float* h_img, \
	const int& data_style, const std::complex<float>*h_data, const vec2D_DBL & dataNOut, const int& option_alignment, const int& option_phase, const bool& if_hpc, const bool& if_mtrc);


/// <summary>
/// Free allocated memory in CPU and GPU.
/// Destroy pointer.
/// </summary>
/// <param name="h_img"></param>
/// <returns></returns>
extern "C" DLL_EXPORT_API void imagingMemDest(float** h_img);

#endif // !ISAR_IMAGING_EXPORT_H_

