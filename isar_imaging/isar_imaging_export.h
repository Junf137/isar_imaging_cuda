#ifndef ISAR_IMAGING_EXPORT_H_
#define ISAR_IMAGING_EXPORT_H_

#include <string>

/// ---* This header file export main imaging function in c++ API *---

#define DLL_EXPORT_API __declspec(dllexport)  // [todo] dllexport ? dllimport

extern "C" DLL_EXPORT_API int isar_imaging(const std::string & dir_path, int sampling_stride, int window_head, int window_len, int data_style, int option_alignment, int option_phase, bool if_hpc, bool if_mtrc);

#endif // !ISAR_IMAGING_EXPORT_H_

