#pragma once
#include<iostream>
#include<vector>
#include<tbb/tbb.h>
#include<tbb/blocked_range.h>
#include<tbb/parallel_for.h>

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <complex>
#include <numeric>
//利用MKL运算库
#include <mkl.h>
//常用的一些参数
#include "Common.cuh"

void MiniEntropyComp(MKL_Complex8* indta, MKL_Complex8* outdta, RadarParameters& mPara);

void FastKTA(MKL_Complex8* indta, MKL_Complex8* outDta, RadarParameters& mPara);

void Hamming(int winLen, float* winVec);

MKL_Complex8 mSum(MKL_Complex8* pAdr, int mLen);

float mSumf(float* pAdr, int mLen);

void DopplerProcessA(MKL_Complex8* indta, float* OutImg, RadarParameters& mPara);