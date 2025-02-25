﻿#include "isar_main.cuh"

int ISAR_RD_Imaging_Main_Ku(float* h_img, cuComplex* d_data, cuComplex* d_data_cut, \
	const RadarParameters& paras, const CUDAHandle& handles, const DATA_TYPE& data_type, const int& option_alignment, const int& option_phase, const bool& if_hrrp, const bool& if_hpc, const bool& if_mtrc)
{
	dim3 block(DEFAULT_THREAD_PER_BLOCK);
	float scale_ifft = 1.0f / static_cast<float>(paras.range_num);


#ifdef DATA_WRITE_BACK_DATAW
	ioOperation::dataWriteBack(INTERMEDIATE_DIR + "dataW.dat", d_data, paras.data_num);
#endif // DATA_WRITE_BACK_DATAW


	if (data_type == DATA_TYPE::IFDS) {
		// ifftshift
		ifftshiftRows << <dim3(((paras.range_num / 2) + block.x - 1) / block.x, paras.echo_num), block >> > (d_data, paras.range_num);
		checkCudaErrors(cudaDeviceSynchronize());

		// ifft
		checkCudaErrors(cufftExecC2C(handles.plan_all_echo_c2c, d_data, d_data, CUFFT_INVERSE));
		checkCudaErrors(cublasCsscal(handles.handle, paras.data_num, &scale_ifft, d_data, 1));

		// ifftshift
		ifftshiftRows << <dim3(((paras.range_num / 2) + block.x - 1) / block.x, paras.echo_num), block >> > (d_data, paras.range_num);
		checkCudaErrors(cudaDeviceSynchronize());
	}


	/******************************
	* HPC
	******************************/
	if ((if_hpc == true) && (data_type == DATA_TYPE::STRETCH)) {
#ifdef SEPARATE_TIMING_
		std::cout << "---* Starting HPC *---\n";
		auto t_hpc_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMING_

		// * Starting HPC
		highSpeedCompensation(d_data, d_velocity, paras, handles);

#ifdef SEPARATE_TIMING_
		auto t_hpc_2 = std::chrono::high_resolution_clock::now();
		std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_hpc_2 - t_hpc_1).count() << "ms\n";
		std::cout << "---* HPC Over *---\n";
		std::cout << "************************************\n\n";
#endif // SEPARATE_TIMING_
	}


#ifdef DATA_WRITE_BACK_HPC
	ioOperation::dataWriteBack(INTERMEDIATE_DIR + "hpc.dat", d_data, paras.data_num);
#endif // DATA_WRITE_BACK_HPC


	/******************
	 * HRRP
	 ******************/
	if (if_hrrp == true) {
#ifdef SEPARATE_TIMING_
		std::cout << "---* Starting Get HRRP *---\n";
		auto t_hrrp_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMING_

		// * HRRP - High Resolution Range Profile.
		getHRRP(d_hrrp, d_data, d_hamming, paras, data_type, handles);

#ifdef SEPARATE_TIMING_
		auto t_hrrp_2 = std::chrono::high_resolution_clock::now();
		std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_hrrp_2 - t_hrrp_1).count() << "ms\n";
		std::cout << "---* Get HRRP Over *---\n";
		std::cout << "************************************\n\n";
#endif // SEPARATE_TIMING_


#ifdef DATA_WRITE_BACK_HRRP
		ioOperation::dataWriteBack(INTERMEDIATE_DIR + "hrrp.dat", d_hrrp, paras.data_num);
#endif // DATA_WRITE_BACK_HRRP
	}


	/******************
	 * Range Alignment and HRRP Centering
	 ******************/
#ifdef SEPARATE_TIMING_
	std::cout << "---* Starting Range Alignment *---\n";
	auto t_ra_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMING_

	// * Range Alignment
	rangeAlignmentParallel(d_data, d_hamming, paras, handles);

#ifdef SEPARATE_TIMING_
	auto t_ra_2 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMING_

	// * Centering HRRP
	int inter_length = 30;
	HRRPCenter(d_data, inter_length, paras, handles);

#ifdef DATA_WRITE_BACK_RA
	ioOperation::dataWriteBack(INTERMEDIATE_DIR + "ra.dat", d_data, paras.data_num);
#endif // DATA_WRITE_BACK_RA

	// * Cutting range profile
	cutRangeProfile(d_data, d_data_cut, paras.range_num, paras.range_num_cut, paras.data_num_cut, handles.handle);

#ifdef SEPARATE_TIMING_
	auto t_ra_3 = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption(aligning)] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_ra_2 - t_ra_1).count() << "ms\n";
	std::cout << "[Time consumption(centering and cutting)] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_ra_3 - t_ra_2).count() << "ms\n";
	std::cout << "---* Range Alignment Over *---\n";
	std::cout << "************************************\n\n";
#endif // SEPARATE_TIMING_


	/**********************
	 * Phase Compensation(Fast_Entropy)
	 **********************/
#ifdef SEPARATE_TIMING_
	std::cout << "---* Starting Phase Compensation *---\n";
	auto t_pc_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMING_

	//int iteration_num = 50;
	//auto t_ra_1 = std::chrono::high_resolution_clock::now();
	//auto t_ra_2 = std::chrono::high_resolution_clock::now();
	//int total_time = 0;
	//for (int i = 0; i < iteration_num; ++i) {
	//	t_ra_1 = std::chrono::high_resolution_clock::now();
	//	// * Profiling operation
	//	fastEntropy(d_data_cut, paras.echo_num, paras.range_num, handles);
	//	t_ra_2 = std::chrono::high_resolution_clock::now();

	//	total_time += static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(t_ra_2 - t_ra_1).count());
	//}
	//total_time /= iteration_num;
	//// print total time
	//std::cout << "total time: " << total_time << " us" << std::endl;

	// * Fast Entropy
	fastEntropy(d_data_cut, paras.echo_num, paras.range_num_cut, handles);

#ifdef SEPARATE_TIMING_
	auto t_pc_2 = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_pc_2 - t_pc_1).count() << "ms\n";
	std::cout << "---* Phase Compensation Over *---\n";
	std::cout << "************************************\n\n";
#endif // SEPARATE_TIMING_


#ifdef DATA_WRITE_BACK_PC
	ioOperation::dataWriteBack(INTERMEDIATE_DIR + "pc.dat", d_data_cut, paras.data_num_cut);
#endif // DATA_WRITE_BACK_PC


	/**********************
	 * MTRC (Migration Through Range Cell)
	 **********************/
	if (if_mtrc == true) {
#ifdef SEPARATE_TIMING_
		std::cout << "---* Starting MTRC *---\n";
		auto t_mtrc_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMING_

		// * MTRC Correction
		mtrc(d_data_cut, paras, data_type, handles);

#ifdef SEPARATE_TIMING_
		auto t_mtrc_2 = std::chrono::high_resolution_clock::now();
		std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_mtrc_2 - t_mtrc_1).count() << "ms\n";
		std::cout << "---* MTRC Over *---\n";
		std::cout << "************************************\n\n";
#endif // SEPARATE_TIMING_
	}


#ifdef DATA_WRITE_BACK_MTRC
	ioOperation::dataWriteBack(INTERMEDIATE_DIR + "mtrc.dat", d_data_cut, paras.data_num_cut);
#endif // DATA_WRITE_BACK_MTRC


	/**********************
	* Post Processing and imaging
	**********************/
#ifdef SEPARATE_TIMING_
	std::cout << "---* Starting Post Processing *---\n";
	auto t_post_processing_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMING_

	// adding hamming
	diagMulMat << <(paras.data_num_cut + block.x - 1) / block.x, block >> > (d_hamming_echoes, d_data_cut, d_data_cut, paras.echo_num, paras.range_num_cut);
	checkCudaErrors(cudaDeviceSynchronize());

	// * Applying fft on each range along the second dimension
	checkCudaErrors(cufftExecC2C(handles.plan_all_range_c2c, d_data_cut, d_data_cut, CUFFT_FORWARD));
	// fftshift
	ifftshiftCols << <dim3(paras.range_num_cut, ((paras.echo_num / 2) + block.x - 1) / block.x), block >> > (d_data_cut, paras.echo_num);
	checkCudaErrors(cudaDeviceSynchronize());

	if (data_type == DATA_TYPE::IFDS) {
		flipud << <dim3(paras.range_num_cut, (paras.echo_num / 2 + block.x - 1) / block.x), block >> > (d_data_cut, paras.echo_num);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	// * final img data
	elementwiseAbs << <(paras.data_num_cut + block.x - 1) / block.x, block >> > (d_data_cut, d_img, paras.data_num_cut);
	checkCudaErrors(cudaDeviceSynchronize());

	// * img (device -> host)
	checkCudaErrors(cudaMemcpy(h_img, d_img, sizeof(float) * paras.data_num_cut, cudaMemcpyDeviceToHost));

#ifdef SEPARATE_TIMING_
	auto t_post_processing_2 = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_post_processing_2 - t_post_processing_1).count() << "ms\n";
	std::cout << "---* Post Processing Over *---\n";
	std::cout << "************************************\n\n";
#endif // SEPARATE_TIMING_


#ifdef DATA_WRITE_BACK_FINAL
	ioOperation::writeFile(INTERMEDIATE_DIR + "final.dat", h_img, paras.data_num_cut);
#endif // DATA_WRITE_BACK_FINAL


	return EXIT_SUCCESS;
}
