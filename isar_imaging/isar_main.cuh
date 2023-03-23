#ifndef ISAR_MAIN_H_
#define ISAR_MAIN_H_

#include "common.cuh"
#include "high_speed_compensation.cuh"
#include "range_alignment.cuh"
#include "phase_adjustment.cuh"

// function[HRRPData, alignData, compData, ISARImageData] = ISAR_RD_Imaging_Main_Ku(Datastyle, DataW, DataNOut, RadarParameters, optionAligment, optionAPhase, IfHighspeedMotionComp, IfRMTRC)
/// <summary>
/// 
/// </summary>
/// <param name="Datastyle"></param>
/// <param name="DataW"> 雷达宽带数据, 直采数据时, 为雷达一维距离像 </param>
/// <param name="DataNOut"></param>
/// <param name="paras"></param>
/// <param name="optionAligment"> range alignment method </param>
/// <param name="optionAPhase"> phase adjustment method </param>
/// <param name="ifHPC"></param>
/// <param name="ifMTRC"></param>
/// <returns></returns>
int ISAR_RD_Imaging_Main_Ku(RadarParameters& paras, const int& datastyle, const vec1D_COM_FLOAT& dataW, const vec2D_FLOAT& dataNOut, const int& optionAligment, const int& optionAPhase, const bool& ifHPC, const bool& ifMTRC);


#endif // ISAR_MAIN_H_
