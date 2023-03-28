#ifndef ISAR_MAIN_H_
#define ISAR_MAIN_H_

#include "common.cuh"
#include "high_speed_compensation.cuh"
#include "range_alignment.cuh"
#include "phase_adjustment.cuh"

/// <summary>
/// 
/// </summary>
/// <param name="paras"></param>
/// <param name="data_style"></param>
/// <param name="dataW"> radar data </param>
/// <param name="dataNOut"></param>
/// <param name="option_alignment"> range alignment method </param>
/// <param name="option_phase"> phase adjustment method </param>
/// <param name="if_hpc"></param>
/// <param name="if_mtrc"></param>
/// <returns></returns>
int ISAR_RD_Imaging_Main_Ku(RadarParameters& paras, const int& data_style, const vec1D_COM_FLOAT& dataW, const vec2D_FLOAT& dataNOut, const int& option_alignment, const int& option_phase, const bool& if_hpc, const bool& if_mtrc);


#endif // ISAR_MAIN_H_
