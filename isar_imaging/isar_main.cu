#include "isar_main.cuh"

int ISAR_RD_Imaging_Main_Ku(RadarParameters& paras, const int& data_style, const vec1D_COM_FLT& dataW, const vec2D_DBL& dataNOut, const int& option_alignment, const int& option_phase, const bool& if_hpc, const bool& if_mtrc)
{
	/******************************
	* Init GPU Device
	******************************/
	// pick the device with highest Gflops/s. (single GPU mode)
	int dev = findCudaDevice(0, static_cast<const char**>(nullptr));
	if (dev == -1) {
		return EXIT_FAILURE;
	}


	/******************************
	* GPU Memory Initialization
	******************************/
	std::cout << "---* Starting GPU Memory Initialization *---\n";
	auto t_init_gpu_1 = std::chrono::high_resolution_clock::now();

	const std::complex<float>* h_data = dataW.data();

	// * Overall cuBlas and cuFFT handle
	CUDAHandle handles(paras.echo_num, paras.range_num);

	// * Overall kernel function configuration
	dim3 block(DEFAULT_THREAD_PER_BLOCK);  // block size

	// * GPU memory allocation
	cuComplex* d_data = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(cuComplex) * paras.data_num));
	checkCudaErrors(cudaMemcpy(d_data, h_data, sizeof(cuComplex) * paras.data_num, cudaMemcpyHostToDevice));  // data (host -> device)

	auto t_init_gpu_2 = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_init_gpu_2 - t_init_gpu_1).count() << "ms\n";
	std::cout << "---* GPU Memory Initialization Over *---\n";
	std::cout << "************************************\n\n";


#ifdef DATA_WRITE_BACK_DATAW
	ioOperation::dataWriteBack(std::string(DIR_PATH) + "dataW.dat", d_data, paras.data_num);
#endif // DATA_WRITE_BACK_DATAW


	/******************************
	* HPC
	******************************/
	if (if_hpc == true) {
		std::cout << "---* Starting HPC *---\n";

		// * Retrieving Velocity Data
		double* h_velocity = new double[paras.echo_num];
		std::transform(dataNOut.cbegin(), dataNOut.cend(), h_velocity, [](const std::vector<double>& v) {return v[1]; });

		double* d_velocity = nullptr;
		checkCudaErrors(cudaMalloc((void**)&d_velocity, sizeof(double) * paras.echo_num));
		checkCudaErrors(cudaMemcpy(d_velocity, h_velocity, sizeof(double) * paras.echo_num, cudaMemcpyHostToDevice));

		auto t_hpc_1 = std::chrono::high_resolution_clock::now();

		// * Starting HPC
		highSpeedCompensation(d_data, d_velocity, paras, handles);

		auto t_hpc_2 = std::chrono::high_resolution_clock::now();

		delete[] h_velocity;
		h_velocity = nullptr;
		checkCudaErrors(cudaFree(d_velocity));
		d_velocity = nullptr;

		std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::microseconds>(t_hpc_2 - t_hpc_1).count() << "us\n";
		std::cout << "---* HPC Over *---\n";
		std::cout << "************************************\n\n";
	}


#ifdef DATA_WRITE_BACK_HPC
	ioOperation::dataWriteBack(std::string(DIR_PATH) + "hpc.dat", d_data, paras.data_num);
#endif // DATA_WRITE_BACK_HPC


	/******************
	 * HRRP
	 ******************/
	std::cout << "---* Starting Get HRRP *---\n";

	// * Adding Hamming Window
	float* d_hamming = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_hamming, sizeof(float) * paras.range_num));
	genHammingVec << <dim3((paras.range_num + block.x - 1) / block.x), block >> > (d_hamming, paras.range_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// d_data = d_data .* repmat(d_hamming, echo_num, 1)
	elementwiseMultiplyRep << <dim3((paras.data_num + block.x - 1) / block.x), block >> > (d_hamming, d_data, d_data, paras.range_num, paras.data_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// * HRRP - High Resolution Range Profile.
	auto t_hrrp_1 = std::chrono::high_resolution_clock::now();

	// d_hrrp = fftshift(fft(d_data))
	cuComplex* d_hrrp = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_hrrp, sizeof(cuComplex) * paras.data_num));
	getHRRP(d_hrrp, d_data, paras, handles);

	auto t_hrrp_2 = std::chrono::high_resolution_clock::now();

	std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_hrrp_2 - t_hrrp_1).count() << "ms\n";
	std::cout << "---* Get HRRP Over *---\n";
	std::cout << "************************************\n\n";


#ifdef DATA_WRITE_BACK_HRRP
	ioOperation::dataWriteBack(std::string(DIR_PATH) + "hrrp.dat", d_hrrp, paras.data_num);
#endif // DATA_WRITE_BACK_HRRP


	/******************
	 * Range Alignment and HRRP Centering
	 ******************/
	std::cout << "---* Starting Range Alignment *---\n";
	auto t_ra_1 = std::chrono::high_resolution_clock::now();

	// * Range Alignment
	rangeAlignmentParallel(d_data, d_hamming, paras, handles);

	auto t_ra_2 = std::chrono::high_resolution_clock::now();

	// * Centering HRRP
	int inter_length = 30;
	HRRPCenter(d_data, inter_length, paras, handles);

#ifdef DATA_WRITE_BACK_RA
	ioOperation::dataWriteBack(std::string(DIR_PATH) + "ra.dat", d_data, paras.data_num);
#endif // DATA_WRITE_BACK_RA

	// * Cutting range profile
	cutRangeProfile(d_data, paras, RANGE_NUM_CUT, handles);

	auto t_ra_3 = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption(aligning)] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_ra_2 - t_ra_1).count() << "ms\n";
	std::cout << "[Time consumption(centering and cutting)] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_ra_3 - t_ra_2).count() << "ms\n";
	std::cout << "---* Range Alignment Over *---\n";
	std::cout << "************************************\n\n";



	/**********************
	 * Phase Compensation
	 * Doppler_Tracking -> RangeVariantPhaseComp -> Fast_Entropy
	 **********************/
	std::cout << "---* Starting Phase Compensation *---\n";
	auto t_pc_1 = std::chrono::high_resolution_clock::now();

	//// * Retrieving Azimuth and Pitch Data
	//double* h_azimuth = new double[paras.echo_num];
	//double* h_pitch = new double[paras.echo_num];
	//std::transform(dataNOut.cbegin(), dataNOut.cend(), h_azimuth, [](std::vector<double> v) {return v[2]; });
	//std::transform(dataNOut.cbegin(), dataNOut.cend(), h_pitch, [](std::vector<double> v) {return v[3]; });

	//// * Range Variant Phase Compensation [todo] optional
	//rangeVariantPhaseComp(d_data, h_azimuth, h_pitch, paras, handles);

	//delete[] h_azimuth;
	//delete[] h_pitch;
	//h_pitch = nullptr;
	//h_azimuth = nullptr;

	auto t_pc_2 = std::chrono::high_resolution_clock::now();

	// * Fast Entropy
	fastEntropy(d_data, paras.echo_num, paras.range_num, handles);

	auto t_pc_3 = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::microseconds>(t_pc_2 - t_pc_1).count() << "us\n";
	std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::microseconds>(t_pc_3 - t_pc_2).count() << "us\n";
	std::cout << "---* Phase Compensation Over *---\n";
	std::cout << "************************************\n\n";


	/**********************
	 * MTRC (Migration Through Range Cell)
	 **********************/
	if (if_mtrc == true) {
		std::cout << "---* Starting MTRC *---\n";
		auto t_mtrc_1 = std::chrono::high_resolution_clock::now();

		// * MTRC Correction
		mtrc(d_data, paras, handles);

		auto t_mtrc_2 = std::chrono::high_resolution_clock::now();
		std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_mtrc_2 - t_mtrc_1).count() << "ms\n";
		std::cout << "---* MTRC Over *---\n";
		std::cout << "************************************\n\n";
	}


	/**********************
	* Post Processing
	**********************/
	//// applying fft on each range along the second dimension
	//checkCudaErrors(cufftExecC2C(handles.plan_all_range_c2c, d_data, d_data, CUFFT_FORWARD));
	//// fftshift
	//ifftshiftCols << <dim3(paras.range_num, ((paras.echo_num / 2) + block.x - 1) / block.x), block >> > (d_data, paras.echo_num);
	//checkCudaErrors(cudaDeviceSynchronize());


	/**********************
	* Final Data Write Back
	**********************/
#ifdef DATA_WRITE_BACK_FINAL
	std::cout << "---* Starting Data Write Back *---\n";

	auto t_data_write_back_1 = std::chrono::high_resolution_clock::now();

	ioOperation::dataWriteBack(std::string(DIR_PATH) + "final.dat", d_data, paras.data_num);
	
	auto t_data_write_back_2 = std::chrono::high_resolution_clock::now();
	
	std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_data_write_back_2 - t_data_write_back_1).count() << "ms\n";
	std::cout << "---* Data Write Back Over *---\n";
	std::cout << "************************************\n\n";
#endif // DATA_WRITE_BACK_FINAL



	/**********************
	* Free Allocated Memory & Destroy Pointer
	**********************/
	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_hamming));
	checkCudaErrors(cudaFree(d_hrrp));

	return EXIT_SUCCESS;
}
