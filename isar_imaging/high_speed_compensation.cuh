#ifndef HIGH_SPEED_COMPENSATION_H_
#define HIGH_SPEED_COMPENSATION_H_

#include "common.cuh"

/// <summary>
/// High Speed Motion Compensation.
/// compensation pahse: fai = 2*pi*K*2*v_tmp/c*tk.^2
/// compensation function : exp(1j * fai)
/// </summary>
/// <param name="d_data">  </param>
/// <param name="Fs"> sampling frequency </param>
/// <param name="band_width"> band width </param>
/// <param name="Tp"> pulse width </param>
/// <param name="h_velocity"> velocity in slow time </param>
/// <param name="echo_num"> </param>
/// <param name="range_num"> </param>
/// <param name="handle"> cublas handle </param>
void highSpeedCompensation(cuComplex* d_data, int Fs, long long band_width, float Tp, float* d_velocity, int echo_num, int range_num, cublasHandle_t handle);


/// <summary>
/// tk2 = ([0:N-1]/fs).^2
/// </summary>
/// <param name="tk2"></param>
/// <param name="Fs"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void genTk2Vec(float* tk2, float Fs, int len);


#endif // !HIGH_SPEED_COMPENSATION_H_