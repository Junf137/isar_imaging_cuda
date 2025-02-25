﻿#ifndef ISAR_MAIN_H_
#define ISAR_MAIN_H_

#include "common.cuh"
#include "high_speed_compensation.cuh"
#include "range_alignment.cuh"
#include "phase_adjustment.cuh"
#include "mtrc.cuh"


extern std::string INTERMEDIATE_DIR;
extern double* d_velocity;
extern float* d_hamming;
extern cuComplex* d_hrrp;
extern float* d_hamming_echoes;
extern float* d_img;


/// <summary>
///
/// </summary>
/// <param name="h_img"></param>
/// <param name="d_data"></param>
/// <param name="d_data_cut"></param>
/// <param name="d_velocity"></param>
/// <param name="d_hamming"></param>
/// <param name="d_hrrp"></param>
/// <param name="d_hamming_echoes"></param>
/// <param name="d_img"></param>
/// <param name="paras"></param>
/// <param name="handles"></param>
/// <param name="data_type"></param>
/// <param name="option_alignment"></param>
/// <param name="option_phase"></param>
/// <param name="if_hrrp"></param>
/// <param name="if_hpc"></param>
/// <param name="if_mtrc"></param>
/// <returns></returns>
int ISAR_RD_Imaging_Main_Ku(float* h_img, cuComplex* d_data, cuComplex* d_data_cut, \
	const RadarParameters& paras, const CUDAHandle& handles, const DATA_TYPE& data_type, const int& option_alignment, const int& option_phase, const bool& if_hrrp, const bool& if_hpc, const bool& if_mtrc);

#endif // ISAR_MAIN_H_
