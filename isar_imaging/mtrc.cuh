#ifndef MTRC_H_
#define MTRC_H_

#include "common.cuh"


extern cuComplex* g_d_range_num_com_flt_1;
extern cuComplex* g_d_range_num_cut_com_flt_1;
extern cuComplex* g_d_echo_num_com_flt_1;
extern cuComplex* g_d_data_num_com_flt_1;
extern cuComplex* g_d_data_num_cut_com_flt_1;
extern cuComplex* g_d_hlf_data_num_com_flt_1;
extern float* g_d_range_num_flt_1;
extern float* g_d_range_num_cut_flt_1;
extern float* g_d_range_num_cut_flt_2;
extern float* g_d_echo_num_flt_1;
extern float* g_d_data_num_cut_flt_1;
extern float* g_d_data_num_flt_1;
extern float* g_d_data_num_flt_2;


/// <summary>
/// 
/// </summary>
/// <param name="d_data"></param>
/// <param name="paras"></param>
/// <param name="data_type"></param>
/// <param name="handles"></param>
void mtrc(cuComplex* d_data, const RadarParameters& paras, const DATA_TYPE& data_type, const CUDAHandle& handles);


/// <summary>
/// Kernel configuration requirement:
/// (1) block_number == {paras.rang_num, (fft_len + block.x - 1) / block.x}
/// (2) thread_per_block == {256, ...}
/// </summary>
/// <param name="d_ww"></param>
/// <param name="d_w"></param>
/// <param name="ww_len"></param>
/// <returns></returns>
__global__ void genWW(cuComplex* d_ww, cuComplex* d_w, int echo_num, int range_num, int ww_len, int fft_len);


/// <summary>
/// Kernel configuration requirement:
/// (1) block_number == {paras.rang_num, (fft_len + block.x - 1) / block.x}
/// (2) thread_per_block == {256, ...}
/// </summary>
/// <param name="d_y"></param>
/// <param name="d_ww"></param>
/// <param name="echo"></param>
/// <param name="echo_num"></param>
/// <param name="ww_len"></param>
/// <param name="fft_len"></param>
/// <returns></returns>
__global__ void gety(cuComplex* d_y, cuComplex* d_a, cuComplex* d_ww, cuComplex* d_data, int echo_num, int range_num, int y_len, int fft_len);


/// <summary>
/// Kernel configuration requirement:
/// (1) block_number == {paras.rang_num, (paras.echo_num + block.x - 1) / block.x}
/// (2) thread_per_block == {256, ...}
/// </summary>
/// <param name="d_czt"></param>
/// <param name="d_ifft"></param>
/// <param name="d_ww"></param>
/// <returns></returns>
__global__ void getCZTOut(cuComplex* d_czt, cuComplex* d_ifft, cuComplex* d_ww, int echo_num);


/// <summary>
/// Applying Chirp-z transform transformation to each range.
/// The element of each separate Z-Transform equal to the number of element in each range.
/// </summary>
/// <param name="d_data"> echo_num * range_num </param>
/// <param name="echo_num"></param>
/// <param name="range_num"></param>
/// <param name="d_w"> vector of length range_num </param>
/// <param name="d_a"> vector of length range_num </param>
void cztRange(cuComplex* d_czt, cuComplex* d_data, cuComplex* d_w, cuComplex* d_a, const int& echo_num, const int& range_num, const CUDAHandle& handles);


/// <summary>
/// 
/// </summary>
/// <param name="d_w"></param>
/// <param name="d_a"></param>
/// <param name="echo_num"></param>
/// <param name="range_num"></param>
/// <param name="constant"> K * 0.5 * T_ref / f0 </param>
/// <param name="posa"></param>
/// <returns></returns>
__global__ void getWandA(cuComplex* d_w, cuComplex* d_a, int echo_num, int range_num, float constant, float posa);

#endif // !MTRC_H_
