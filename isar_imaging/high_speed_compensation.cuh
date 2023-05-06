﻿#ifndef HIGH_SPEED_COMPENSATION_H_
#define HIGH_SPEED_COMPENSATION_H_

#include "common.cuh"

extern std::string INTERMEDIATE_DIR;

/// <summary>
/// High Speed Motion Compensation.
/// compensation phase: fai = 2*pi*K*2*v_tmp/c*tk.^2
/// compensation function : exp(1j * fai)
/// </summary>
/// <param name="d_data"></param>
/// <param name="d_velocity"></param>
/// <param name="paras"></param>
/// <param name="handles"></param>
void highSpeedCompensation(cuDoubleComplex* d_data, double* d_velocity, const RadarParameters& paras, const CUDAHandle& handles);


/// <summary>
/// tk2 = ([0:N-1]/fs).^2
/// </summary>
/// <param name="d_tk2"></param>
/// <param name="Fs"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void genTk2Vec(double* d_tk2, double Fs, int len);


#endif // !HIGH_SPEED_COMPENSATION_H_