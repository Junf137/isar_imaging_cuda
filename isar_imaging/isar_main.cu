#include "isar_main.cuh"

int ISAR_RD_Imaging_Main_Ku(float* h_img, cuComplex* d_data, cuComplex* d_data_cut, double* d_velocity, float* d_hamming, cuComplex* d_hrrp, float* d_hamming_echoes, float* d_img, \
	RadarParameters& paras, const CUDAHandle& handles, const int& data_style, const std::complex<float>* h_data, const vec2D_DBL& dataNOut, \
	const int& option_alignment, const int& option_phase, const bool& if_hpc, const bool& if_mtrc)
{
	dim3 block(DEFAULT_THREAD_PER_BLOCK);

	// d_data (host -> device)
	checkCudaErrors(cudaMemcpy(d_data, h_data, sizeof(cuComplex) * paras.data_num, cudaMemcpyHostToDevice));


#ifdef DATA_WRITE_BACK_DATAW
	ioOperation::dataWriteBack(std::string(DIR_PATH) + "dataW.dat", d_data, paras.data_num);
#endif // DATA_WRITE_BACK_DATAW


	/******************************
	* HPC
	******************************/
	if (if_hpc == true) {
#ifdef SEPARATE_TIMEING_
		std::cout << "---* Starting HPC *---\n";
		auto t_hpc_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

		// * Retrieving Velocity Data
		double* h_velocity = new double[paras.echo_num];
		std::transform(dataNOut.cbegin(), dataNOut.cend(), h_velocity, [](const vec1D_DBL& v) {return v[1]; });
		checkCudaErrors(cudaMemcpy(d_velocity, h_velocity, sizeof(double) * paras.echo_num, cudaMemcpyHostToDevice));

		// * Starting HPC
		highSpeedCompensation(d_data, d_velocity, paras, handles);

		delete[] h_velocity;
		h_velocity = nullptr;

#ifdef SEPARATE_TIMEING_
		auto t_hpc_2 = std::chrono::high_resolution_clock::now();
		std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_hpc_2 - t_hpc_1).count() << "ms\n";
		std::cout << "---* HPC Over *---\n";
		std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_
	}


#ifdef DATA_WRITE_BACK_HPC
	ioOperation::dataWriteBack(std::string(DIR_PATH) + "hpc.dat", d_data, paras.data_num);
#endif // DATA_WRITE_BACK_HPC


	/******************
	 * HRRP
	 ******************/
#ifdef SEPARATE_TIMEING_
	std::cout << "---* Starting Get HRRP *---\n";
	auto t_hrrp_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

	// * Adding hamming window in range dimension
	genHammingVec << <dim3((paras.range_num + block.x - 1) / block.x), block >> > (d_hamming, paras.range_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// d_data = d_data .* repmat(d_hamming, echo_num, 1)
	elementwiseMultiplyRep << <dim3((paras.data_num + block.x - 1) / block.x), block >> > (d_hamming, d_data, d_data, paras.range_num, paras.data_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// * HRRP - High Resolution Range Profile.
	// d_hrrp = fftshift(fft(d_data))
	getHRRP(d_hrrp, d_data, paras, handles);

#ifdef SEPARATE_TIMEING_
	auto t_hrrp_2 = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_hrrp_2 - t_hrrp_1).count() << "ms\n";
	std::cout << "---* Get HRRP Over *---\n";
	std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_


#ifdef DATA_WRITE_BACK_HRRP
	ioOperation::dataWriteBack(std::string(DIR_PATH) + "hrrp.dat", d_hrrp, paras.data_num);
#endif // DATA_WRITE_BACK_HRRP


	/******************
	 * Range Alignment and HRRP Centering
	 ******************/
#ifdef SEPARATE_TIMEING_
	std::cout << "---* Starting Range Alignment *---\n";
	auto t_ra_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

	// * Range Alignment
	rangeAlignmentParallel(d_data, d_hamming, paras, handles);

#ifdef SEPARATE_TIMEING_
	auto t_ra_2 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

	// * Centering HRRP
	int inter_length = 30;
	HRRPCenter(d_data, inter_length, paras, handles);

#ifdef DATA_WRITE_BACK_RA
	ioOperation::dataWriteBack(std::string(DIR_PATH) + "ra.dat", d_data, paras.data_num);
#endif // DATA_WRITE_BACK_RA

	// * Cutting range profile
	cutRangeProfile(d_data_cut, d_data, paras, RANGE_NUM_CUT, handles);

#ifdef SEPARATE_TIMEING_
	auto t_ra_3 = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption(aligning)] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_ra_2 - t_ra_1).count() << "ms\n";
	std::cout << "[Time consumption(centering and cutting)] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_ra_3 - t_ra_2).count() << "ms\n";
	std::cout << "---* Range Alignment Over *---\n";
	std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_



	/**********************
	 * Phase Compensation
	 * Doppler_Tracking -> RangeVariantPhaseComp -> Fast_Entropy
	 **********************/
#ifdef SEPARATE_TIMEING_
	std::cout << "---* Starting Phase Compensation *---\n";
	auto t_pc_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

	//// * Retrieving Azimuth and Pitch Data
	//double* h_azimuth = new double[paras.echo_num];
	//double* h_pitch = new double[paras.echo_num];
	//std::transform(dataNOut.cbegin(), dataNOut.cend(), h_azimuth, [](vec1D_DBL v) {return v[2]; });
	//std::transform(dataNOut.cbegin(), dataNOut.cend(), h_pitch, [](vec1D_DBL v) {return v[3]; });

	//// * Range Variant Phase Compensation [todo] optional
	//rangeVariantPhaseComp(d_data_cut, h_azimuth, h_pitch, paras, handles);

	//delete[] h_azimuth;
	//delete[] h_pitch;
	//h_pitch = nullptr;
	//h_azimuth = nullptr;

#ifdef SEPARATE_TIMEING_
	auto t_pc_2 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

	// * Fast Entropy
	fastEntropy(d_data_cut, paras.echo_num, paras.range_num, handles);

#ifdef SEPARATE_TIMEING_
	auto t_pc_3 = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_pc_2 - t_pc_1).count() << "ms\n";
	std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_pc_3 - t_pc_2).count() << "ms\n";
	std::cout << "---* Phase Compensation Over *---\n";
	std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_


#ifdef DATA_WRITE_BACK_PC
	ioOperation::dataWriteBack(std::string(DIR_PATH) + "pc.dat", d_data_cut, paras.data_num);
#endif // DATA_WRITE_BACK_PC


	/**********************
	 * MTRC (Migration Through Range Cell)
	 **********************/
	if (if_mtrc == true) {
#ifdef SEPARATE_TIMEING_
		std::cout << "---* Starting MTRC *---\n";
		auto t_mtrc_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

		// * MTRC Correction
		mtrc(d_data_cut, paras, handles);

#ifdef SEPARATE_TIMEING_
		auto t_mtrc_2 = std::chrono::high_resolution_clock::now();
		std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_mtrc_2 - t_mtrc_1).count() << "ms\n";
		std::cout << "---* MTRC Over *---\n";
		std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_
	}


#ifdef DATA_WRITE_BACK_MTRC
	ioOperation::dataWriteBack(std::string(DIR_PATH) + "mtrc.dat", d_data_cut, paras.data_num);
#endif // DATA_WRITE_BACK_MTRC


	/**********************
	* Post Processing and imaging
	**********************/
#ifdef SEPARATE_TIMEING_
	std::cout << "---* Starting Post Processing *---\n";
	auto t_post_processing_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

	// * Adding hamming window in range dimension
	genHammingVec << <dim3((paras.echo_num + block.x - 1) / block.x), block >> > (d_hamming_echoes, paras.echo_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// adding hamming
	diagMulMat << <(paras.data_num + block.x - 1) / block.x, block >> > (d_hamming_echoes, d_data_cut, d_data_cut, paras.echo_num, paras.range_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// * Applying fft on each range along the second dimension
	checkCudaErrors(cufftExecC2C(handles.plan_all_range_c2c, d_data_cut, d_data_cut, CUFFT_FORWARD));
	// fftshift
	ifftshiftCols << <dim3(paras.range_num, ((paras.echo_num / 2) + block.x - 1) / block.x), block >> > (d_data_cut, paras.echo_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// * final img data
	elementwiseAbs << <(paras.data_num + block.x - 1) / block.x, block >> > (d_data_cut, d_img, paras.data_num);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(h_img, d_img, sizeof(float) * paras.data_num, cudaMemcpyDeviceToHost));  // img (device -> host)

#ifdef SEPARATE_TIMEING_
	auto t_post_processing_2 = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_post_processing_2 - t_post_processing_1).count() << "ms\n";
	std::cout << "---* Post Processing Over *---\n";
	std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_


#ifdef DATA_WRITE_BACK_FINAL
	ioOperation::writeFile(std::string(DIR_PATH) + "final.dat", h_img, paras.data_num);
#endif // DATA_WRITE_BACK_FINAL


	return EXIT_SUCCESS;
}
