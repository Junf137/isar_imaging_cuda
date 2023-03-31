#ifndef MTRC_H_
#define MTRC_H_

#include "common.cuh"


/// <summary>
/// 
/// </summary>
/// <param name="d_data"></param>
/// <param name="paras"></param>
/// <param name="handles"></param>
void mtrc(cuComplex* d_data, const RadarParameters& paras, const CUDAHandle& handles);

#endif // !MTRC_H_
