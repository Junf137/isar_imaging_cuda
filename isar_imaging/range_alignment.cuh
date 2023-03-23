#ifndef RANGEALIGNMENT_H_
#define RANGEALIGNMENT_H_

#include "common.cuh"


/// <summary>
/// Range alignment using linej algorithm
/// </summary>
/// <param name="d_data"></param>
/// <param name="hamming_window"></param>
/// <param name="paras"></param>
/// <param name="handles"></param>
void rangeAlignment(cuComplex* d_data, float* hamming_window, const RadarParameters& paras, const CUDAHandle& handles);


/// <summary>
/// Range alignment using merging parallel algorithm.
/// </summary>
/// <param name="d_data"></param>
/// <param name="hamming_window"></param>
/// <param name="paras"></param>
/// <param name="handles"></param>
void rangeAlignmentParallel(cuComplex* d_data, float* hamming_window, const RadarParameters& paras, const CUDAHandle& handles);


/// <summary>
/// Getting average profile of every stride in time domain using parallel algorithm. Storing the result in the first row of stride.
/// Each block is responsible for calculating the average profile of a stride inside a single column
/// In this algorithm, stride won't be bigger than value of echo_num, so we can assign stride number of thread to each block.
/// Since each column is partitioned into (echo_num / stride) number of blocks, representing the arrangement of block by two-dimension is needed.
/// Kernel configuration requirement:
/// (1) block_number == cols * (echo_num / stride)
/// (2) shared_memory_number == thread_per_block == stride
/// </summary>
/// <param name="d_data"></param>
/// <param name="d_ave_profile"></param>
/// <param name="rows"></param>
/// <param name="cols"></param>
/// <param name="stride"></param>
/// <returns></returns>
__global__ void getAveProfileParallel(float* d_data, float* d_ave_profile, int rows, int cols, const int& stride);


/// <summary>
/// Conjugate multiply between each two stride's fft of average profile, store result in first row of each two stride.
/// Kernel configuration requirement:
/// (1) block_number == cols
/// (2) thread_per_block == (rows / (stride * 2))
/// </summary>
/// <param name="d_data"></param>
/// <param name="rows"></param>
/// <param name="cols"></param>
/// <param name="stride"></param>
/// <returns></returns>
__global__ void conjMulAveProfile(cuComplex* d_data, int rows, int cols, int stride);


/// <summary>
/// Kernel configuration requirement:
/// (1) block_number == rows
/// (2) thread_per_block == {256}
/// </summary>
/// <param name="d_data_l"></param>
/// <param name="d_data_r"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void ifftshiftRows(float* d_data, int cols);


/// <summary>
/// Performing binomial fix.
/// </summary>
/// <param name="d_vec_corr"></param>
/// <param name="maxPos"></param>
/// <returns>fixed value</returns>
__device__ float binomialFixDevice(float* d_vec_corr, int maxPos);


/// <summary>
/// Getting the max element of every single row in matrix d_data(incorporating abs operation).
/// Each block is responsible for the calculation of a single row.
/// Since d_data is arranged by row-major in device memory and element per row is usually much bigger than number of thread per block.
/// Therefore calculating partial maximum in each thread before performing reduction on shared memory in block-scale is necessary.
/// Note that the value of cols should be power of 2.
/// Kernel configuration requirements:
/// (1) block_number == rows
/// (2) shared_memory_number == thread_per_block == {256, 512, 1024}
/// </summary>
/// <param name="d_data"></param>
/// <param name="d_max_idx_rows"></param>
/// <param name="rows"></param>
/// <param name="cols"></param>
/// <returns></returns>
__global__ void maxRowsIdxABS(float* d_data, float* d_max_rows_idx, int rows, int cols);


/// <summary>
/// Generating frequency moving vector for every two stride.
/// Kernel configuration requirements:
/// (1) block_number == {(range_num + block.x - 1) / block.x, echo / (stride * 2)}
/// (2) shared_memory_number == thread_per_block == {256, 512, 1024}
/// </summary>
/// <param name="d_freq_mov_vec"></param>
/// <param name="d_max_idx"></param>
/// <param name="rows"></param>
/// <param name="cols"></param>
/// <returns></returns>
__global__ void genFreqMovParallel(cuComplex* d_freq_mov_vec, float* d_max_idx, int cols, int stride);


/// <summary>
/// Performing element-wise multiply to the right stride of each two strides with frequency moving vector stored in first row of each two stride.
/// Kernel configuration requirements:
/// (1) block_number == {(range_num + block.x - 1) / block.x, stride, echo / (stride * 2)}
/// (2) shared_memory_number == thread_per_block == {256, 512, 1024}
/// </summary>
/// <param name="d_data"></param>
/// <param name="d_freq_mov_vec"></param>
/// <param name="cols"></param>
/// <param name="stride"></param>
/// <returns></returns>
__global__ void alignWithinStride(cuComplex* d_data, cuComplex* d_freq_mov_vec, int cols, int stride);


/// <summary>
/// hamming .* real(exp(-1j*pi*[0:N-1]))
/// </summary>
/// <param name="hamming"></param>
/// <param name="d_freq_centering_vec"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void genFreqCenteringVec(float* hamming, cuComplex* d_freq_centering_vec, int len);


/// <summary>
/// Calculate correlation function using fft algorithm.
/// Two engaged vectors are required to share the same length.
/// </summary>
/// <param name="vec_a"> first vector </param>
/// <param name="vec_b1"> second vector </param>
/// <param name="vec_Corr"> correlation result </param>
void getCorrelation(float* d_vec_corr, float* d_vec_a, float* d_vec_b, int len, cufftHandle plan_one_echo_r2c, cufftHandle plan_one_echo_c2r);


/// <summary>
/// Binomial fixing, get the precise position of max value.
/// </summary>
/// <param name="d_vec_corr"></param>
/// <param name="d_xstar"></param>
/// <param name="maxPos"></param>
/// <returns></returns>
__global__ void binomialFix(float* d_vec_corr, float* d_xstar, int maxPos);


/// <summary>
/// Updating d_vec_b in range alignment iteration.
/// d_vec_b = 0.95f * d_vec_b + abs(d_data(i,:))
/// </summary>
/// <param name="d_vec_b"></param>
/// <param name="d_data_i"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void updateVecB(float* d_vec_b, cuComplex* d_data_i, int len);


/// <summary>
/// d_freq_mov_vec = exp(-1j * 2 * pi * [0:N-1] * mopt / N)
/// </summary>
/// <param name="d_freq_mov_vec"></param>
/// <param name="mopt"></param>
/// <param name="len"></param>
/// <returns></returns>
__global__ void genFreqMovVec(cuComplex* d_freq_mov_vec, float shit_num, int len);


/// <summary>
/// Centering HRRP
/// </summary>
/// <param name="d_data"></param>
/// <param name="paras"></param>
/// <param name="inter_length"></param>
/// <param name="handle"></param>
/// <param name="plan_all_echo_c2c"></param>
void HRRPCenter(cuComplex* d_data, const int& inter_length, const RadarParameters& paras, const CUDAHandle& handles);


/// <summary>
/// slightly slower than doing circshift in frequency domain.
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="d_data"></param>
/// <param name="frag_num"> fragment length </param>
/// <param name="shift"> shift number for each fragment </param>
/// <param name="len"> total length </param>
template <typename T>
void circshift(T* d_data, int frag_len, int shift, int len);


/// <summary>
/// 
/// </summary>
/// <param name="d_in"></param>
/// <param name="d_out"></param>
/// <param name="frag_len"></param>
/// <param name="shift_num"></param>
/// <param name="len"></param>
/// <returns></returns>
template <typename T>
__global__ void circShiftKernel(T* d_in, T* d_out, int frag_len, int shift_num, int len);


/// <summary>
/// doing circshift in frequency domain.
/// </summary>
/// <param name="d_data"></param>
/// <param name="frag_len"></param>
/// <param name="shift"></param>
/// <param name="len"></param>
/// <param name="handle"></param>
/// <param name="plan_all_echo_c2c"></param>
void circshiftFreq(cuComplex* d_data, int frag_len, float shift, int len, cublasHandle_t handle, cufftHandle plan_all_echo_c2c);


/// <summary>
/// GPU核函数，根据索引值附近ARP均值剔除野值. (reference: HRRPCenter.m)
/// todo: optimization
/// </summary>
/// <param name="d_arp_ave"> 待计算的ARP均值向量，长度为indices_length </param>
/// <param name="indices"> indices = find(ARP>low_threshold_gray) </param>
/// <param name="arp"> average profile, length is the number of point in range dimension </param>
/// <param name="indices_length"></param>
/// <param name="WL"> 类似于CFAR中的参考单元 </param>
/// <param name="range_num"></param>
/// <returns></returns>
__global__ void getARPMean(float* d_arp_ave, int* indices, float* arp, int indices_length, int WL, int range_num);

#endif // !RANGEALIGNMENT_H_
