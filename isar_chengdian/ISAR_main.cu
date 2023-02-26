/*
 * 成电项目：ISAR实时成像GPU实现
 * 作者    ：鲁越
 * 起始时间：2019-09-04
 */
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <complex>

#include "cuda_runtime.h"
#include "cufft.h"
#include "cuComplex.h"
#include "cublas_v2.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/complex.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>


#include "Common.cuh"
#include "HighSpeedCompensation.cuh"
#include "RVPCompensation.cuh"
#include "RangeAlignment.cuh"
#include "PhaseAdjustment.cuh"

#include "Net.h"
#include "NetTCP.h"
#include "NetUDP.h"

#include "isarfun.h"

typedef thrust::complex<float> comThr;
const int PORT = 8080;
int main()
{
	/*************************
	 * 数据由磁盘读到内存
	 *************************/
	RadarParameters paras(512, 20000, 6000000000, 35000000512, 0.001, 20000000, 10000);
	//包络对齐后截取长度
	const int range_length = 4096;
	paras.cLen = range_length;
	//YSQ
	ioOperation io;
	std::complex<float>* original_data = new std::complex<float>[paras.num_echoes * paras.num_range_bins]; // 存放时域回波
	std::string data_path("D:\\ACAMEL\\X130Data\\2021年04月20日19时06分19秒(STRETCHDATA)_FS_41579_0G_11\\00001.dat");
	int if_read_success = 0;
	if (if_read_success == 1) {
		system("pause");
		return EXIT_FAILURE;
	}
	//用于高速补偿模块
	float* velocity_data = new float[paras.num_echoes];
	float* range_data = new float[paras.num_echoes];
	float* azimuth_data = new float[paras.num_echoes];
	float* pitch_data = new float[paras.num_echoes];

	// GPU端分配的地址
	// format:(slow-time * fast-time), row major
	cuComplex* d_data;
	checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(cuComplex) * paras.num_echoes * paras.num_range_bins));
	auto tStart_dataTrans = std::chrono::high_resolution_clock::now();
	// 主程序里定义一个cublas的handle，其余地方需要用cublas的都将这里的handle作为参数传入
	cublasHandle_t handle;
	checkCublasErrors(cublasCreate(&handle));

	//读取数据
	if_read_success = io.ReadData(data_path, original_data, velocity_data, range_data, pitch_data, azimuth_data, paras);
	//拷贝到GPU
	checkCudaErrors(cudaMemcpy(d_data, original_data, sizeof(cuComplex) * paras.num_echoes * paras.num_range_bins, cudaMemcpyHostToDevice));
	comThr* thr_d_data_temp = reinterpret_cast<comThr*>(d_data);
	thrust::device_ptr<comThr>thr_data = thrust::device_pointer_cast(thr_d_data_temp);




	std::cout << "Echo Data has been loaded!" << std::endl;
	std::cout << "************************************\n" << std::endl;

	/*********************
	 * 实现高速运动补偿
	 *********************/
	auto tStart_HPC = std::chrono::high_resolution_clock::now();

	HighSpeedCompensation(d_data, paras.Fs, paras.fc, paras.band_width, paras.Tp, velocity_data, range_data, paras.num_echoes, paras.num_range_bins, handle);

	cudaDeviceSynchronize();
	auto tEnd_HPC = std::chrono::high_resolution_clock::now();
	std::cout << "************************************" << std::endl;
	std::cout << "Time consumption of high speed compensation: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_HPC - tStart_HPC).count()
		<< "ms" << std::endl;

	delete[]velocity_data;
	velocity_data = NULL;
	delete[]range_data;
	range_data = NULL;
	std::cout << "High speed compensation has been done!" << std::endl;
	std::cout << "************************************\n" << std::endl;


	// ==== 脉内补偿模块 ==== //
	// *待完成
	//std::string real_data_path2("E:\\vs project\\ISARImaging2020\\data\\data_new\\DIC_I.dat");
	//std::string image_data_path2("E:\\vs project\\ISARImaging2020\\data\\data_new\\DIC_Q.dat");
	//if_read_success = io.ReadFile(real_data_path2, image_data_path2, image_data_path2);
	//if (if_read_success == 1) {
	//	system("pause");
	//	return EXIT_FAILURE;
	//}
	//checkCudaErrors(cudaMemcpy(d_data, original_data, sizeof(cuComplex)*paras.num_echoes*paras.num_range_bins, cudaMemcpyHostToDevice));
	// ====================== //


	/******************
	 * 获取距离像序列
	 ******************/
	std::string arrange_rank = "rowMajor";
	thrust::device_vector<int>shift_vec_hrrp(paras.num_echoes * paras.num_range_bins);
	fftshiftMulWay(shift_vec_hrrp, paras.num_echoes * paras.num_range_bins);
	thrust::transform(thrust::device, thr_data, thr_data + paras.num_echoes * paras.num_range_bins, shift_vec_hrrp.begin(),
		thr_data, []__host__ __device__(thrust::complex<float> x, int y) { return x * float(y); });

	GetHRRP(d_data, paras.num_echoes, paras.num_range_bins, arrange_rank);

	//测试距离像处理是否正确
	/*
	std::complex<float>* test_data = new std::complex<float>[paras.num_echoes * paras.num_range_bins];
	checkCudaErrors(cudaMemcpy(test_data, d_data, sizeof(cuComplex)* paras.num_echoes* paras.num_range_bins, cudaMemcpyDeviceToHost));
	std::cout << "Writing data...\n";
	std::string path_out = "D:\\ACAMEL\\X130Data\\2021年05月13日18时59分54秒(STRETCHDATA)_FS_05398_6G_11\\rangeoutb.dat";
	io.WriteFile(path_out, test_data, paras.num_echoes * paras.num_range_bins);
	delete[] test_data;
	test_data = NULL;
	*/
	/******************
	 * RVP消除
	 * 备注：MATLAB程序中有这一步，
	 *       但实际中应该在GPU处理之前就完成RVP消除，
	 *       这里是否必要还需讨论。
	 ******************/
	auto tStart_RVP = std::chrono::high_resolution_clock::now();

	RVPCompensation(d_data, paras, handle);

	cudaDeviceSynchronize();
	//测试RVP数据
	/*
	std::complex<float>* test_data = new std::complex<float>[paras.num_echoes * paras.num_range_bins];
	checkCudaErrors(cudaMemcpy(test_data, d_data, sizeof(cuComplex)* paras.num_echoes* paras.num_range_bins, cudaMemcpyDeviceToHost));
	std::cout << "Writing data...\n";
	std::string path_out = "D:\\ACAMEL\\X130Data\\2021年05月13日18时59分54秒(STRETCHDATA)_FS_05398_6G_11\\rvpout.dat";
	io.WriteFile(path_out, test_data, paras.num_echoes* paras.num_range_bins);
	delete[] test_data;
	test_data = NULL;
	*/
	auto tEnd_RVP = std::chrono::high_resolution_clock::now();
	std::cout << "************************************" << std::endl;
	std::cout << "Time consumption of RVP compensation: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_RVP - tStart_RVP).count()
		<< "ms" << std::endl;
	std::cout << "RVP compensation has been done!" << std::endl;
	std::cout << "************************************\n" << std::endl;

	/******************
	 * 包络对齐以及距离像序列平移
	 ******************/
	 // 对齐包络
	auto tStart_RA = std::chrono::high_resolution_clock::now();

	RangeAlignment_linej(d_data, paras, shift_vec_hrrp);

	// 将目标移到中心
	unsigned int inter_length = 30;
	HRRPCenter(d_data, paras, inter_length);

	cudaDeviceSynchronize();
	/*
	std::complex<float>* test_data = new std::complex<float>[paras.num_echoes * paras.num_range_bins];
	checkCudaErrors(cudaMemcpy(test_data, d_data, sizeof(cuComplex)* paras.num_echoes* paras.num_range_bins, cudaMemcpyDeviceToHost));
	std::cout << "Writing data...\n";
	std::string path_out = "D:\\ACAMEL\\X130Data\\2021年05月13日18时59分54秒(STRETCHDATA)_FS_05398_6G_11\\centerout.dat";
	io.WriteFile(path_out, test_data, paras.num_echoes* paras.num_range_bins);
	delete[] test_data;
	test_data = NULL;
	*/
	auto tEnd_RA = std::chrono::high_resolution_clock::now();
	std::cout << "************************************" << std::endl;
	std::cout << "Time consumption of range alignment: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_RA - tStart_RA).count()
		<< "ms" << std::endl;
	std::cout << "Range alignment has been done!" << std::endl;
	std::cout << "************************************\n" << std::endl;

	cuComplex* d_data_cut;
	checkCudaErrors(cudaMalloc((void**)&d_data_cut, sizeof(cuComplex) * range_length * paras.num_echoes));  // 数据裁剪后存放于此
	auto tStart_cut = std::chrono::high_resolution_clock::now();

	// YSQ 改为求最大值点，确定开始距离单元
	float* range_abs;                                                         // 求幅度
	checkCudaErrors(cudaMalloc((void**)&range_abs, sizeof(float) * paras.num_range_bins));
	thrust::device_ptr<float>thr_range_abs(range_abs);
	thrust::transform(thrust::device, thr_data, thr_data + paras.num_range_bins, thr_range_abs,
		[]__host__ __device__(thrust::complex<float> x) { return thrust::abs(x); });
	thrust::device_ptr<float> min_ptr = thrust::max_element(thr_range_abs, thr_range_abs + paras.num_range_bins);
	int mPos = &min_ptr[0] - &thr_range_abs[0];
	paras.Pos = mPos;
	cutRangeProfile(d_data, d_data_cut, range_length, paras);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaFree(range_abs));
	auto tEnd_cut = std::chrono::high_resolution_clock::now();
	std::cout << "************************************" << std::endl;
	std::cout << "Time consumption of cutting range profiles: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_cut - tStart_cut).count()
		<< "ms" << std::endl;
	std::cout << "Cut range profiles has been done!" << std::endl;
	std::cout << "************************************\n" << std::endl;
	checkCudaErrors(cudaFree(d_data));
	paras.num_range_bins = range_length;
	//测试截取后的距离像

	/*
	std::complex<float>* test_data = new std::complex<float>[paras.num_echoes * paras.num_range_bins];
	checkCudaErrors(cudaMemcpy(test_data, d_data_cut, sizeof(cuComplex)* paras.num_echoes* paras.num_range_bins, cudaMemcpyDeviceToHost));
	std::cout << "Writing data...\n";
	std::string path_outa = "D:\\ACAMEL\\X130Data\\2021年05月13日18时59分54秒(STRETCHDATA)_FS_05398_6G_11\\rangecutb.dat";
	io.WriteFile(path_outa, test_data, paras.num_echoes* paras.num_range_bins);
	delete[] test_data;
	test_data = NULL;
	*/


	/**********************
	 * 实现各种相位补偿
	 * 多普勒跟踪->距离向空变的相位补偿->快速最小熵
	 **********************/
	auto tStart_droptrace = std::chrono::high_resolution_clock::now();

	Doppler_Tracking(d_data_cut, paras);

	cudaDeviceSynchronize();
	auto tEnd_droptrace = std::chrono::high_resolution_clock::now();
	std::cout << "************************************" << std::endl;
	std::cout << "Time consumption of Doppler centriod tracing: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_droptrace - tStart_droptrace).count()
		<< "ms" << std::endl;
	std::cout << "Doppler centroid tracing has been done!" << std::endl;
	std::cout << "************************************\n" << std::endl;

	auto tStart_ran = std::chrono::high_resolution_clock::now();

	RangeVariantPhaseComp(d_data_cut, paras, azimuth_data, pitch_data);

	cudaDeviceSynchronize();
	auto tEnd_ran = std::chrono::high_resolution_clock::now();
	std::cout << "************************************" << std::endl;
	std::cout << "Time consumption of range variant phase compensation: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_ran - tStart_ran).count()
		<< "ms" << std::endl;
	std::cout << "Range variant phase compensation has been done!" << std::endl;
	std::cout << "************************************\n" << std::endl;

	Fast_Entropy(d_data_cut, paras);
	//cudaDeviceSynchronize();

	//2022/1/21 改为传给CPU进行多核处理
	MKL_Complex8* h_dta_cut = (MKL_Complex8*)MKL_malloc(sizeof(MKL_Complex8) * paras.num_echoes * paras.num_range_bins, 64);
	checkCudaErrors(cudaMemcpy(h_dta_cut, d_data_cut, sizeof(MKL_Complex8) * paras.num_echoes * paras.num_range_bins, cudaMemcpyDeviceToHost));

	MKL_Complex8* h_image_abs = (MKL_Complex8*)MKL_malloc(sizeof(MKL_Complex8) * paras.num_echoes * paras.num_range_bins, 64);

	/* 销毁 */
	mkl_free(h_dta_cut);
	mkl_free(h_phase);
	delete[] original_data;
	original_data = nullptr;
	delete[] velocity_data;
	velocity_data = nullptr;
	delete[] h_image_abs;
	h_image_abs = nullptr;
	checkCudaErrors(cudaFree(d_data_cut));

	checkCublasErrors(cublasDestroy(handle));
	std::cout << "success!" << std::endl;
	system("pause");
	return 0;
}