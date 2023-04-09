#ifndef ISAR_IMAGING_EXPORT_H_
#define ISAR_IMAGING_EXPORT_H_

#include <string>
#include <vector>
#include <complex>


#define DLL_EXPORT_API __declspec(dllexport)  // [todo] dllexport ? dllimport


/// ---* This header file export main imaging function in c++ API using c-style naming standard *---
typedef std::vector<std::vector<int>> vec2D_INT;
typedef std::vector<std::vector<float>> vec2D_FLT;
typedef std::vector<std::vector<double>> vec2D_DBL;
typedef std::vector<std::complex<float>> vec1D_COM_FLT;
typedef std::vector<std::vector<std::complex<float>>> vec2D_COM_FLT;


extern "C" DLL_EXPORT_API int gpuDevInit();

extern "C" DLL_EXPORT_API int dataParsing(vec2D_DBL * dataN, vec2D_INT * stretchIndex, std::vector<float>*turnAngle, \
	const std::string & dir_path);

extern "C" DLL_EXPORT_API int dataExtracting(vec1D_COM_FLT * dataW, vec2D_DBL * dataNOut, \
	const vec2D_DBL & dataN, const vec2D_INT & stretchIndex, const std::vector<float>&turnAngle, const int& sampling_stride, const int& window_head, const int& window_len);

extern "C" DLL_EXPORT_API void imagingMemInit(float*& h_img);

extern "C" DLL_EXPORT_API void isarMainSingle(float* h_img, \
	const int& data_style, const std::complex<float>*h_data, const vec2D_DBL & dataNOut, const int& option_alignment, const int& option_phase, const bool& if_hpc, const bool& if_mtrc);

extern "C" DLL_EXPORT_API void imagingMemDest(float*& h_img);

#endif // !ISAR_IMAGING_EXPORT_H_

