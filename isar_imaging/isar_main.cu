#include "isar_main.cuh"

int ISAR_RD_Imaging_Main_Ku(RadarParameters& paras, const int& datastyle, const vec1D_COM_FLOAT& dataW, const vec2D_FLOAT& dataNOut, const int& optionAligment, const int& optionAPhase, const bool& ifHPC, const bool& ifMTRC) {
	/******************************
	* Init GPU Device
	******************************/
	int devID = 0;  // pick the device with highest Gflops/s. (single GPU mode)
	checkCudaErrors(cudaSetDevice(devID));

	// * CUDA Compability Information
	//int major = 0, minor = 0;
	//checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
	//checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
	//printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, _ConvertSMVer2ArchName(major, minor), major, minor);

	
	/******************************
	* GPU Memory Initialization
	******************************/
	std::cout << "---* Starting GPU Memory Initialization *---\n";
	auto tStart_InitGPU = std::chrono::high_resolution_clock::now();

	const std::complex<float>* h_data = dataW.data();

	int data_num = paras.echo_num * paras.range_num;

	// * Overall cuBlas handle
	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	// * Overall cuFFT plan
	cufftHandle plan_all_echo_c2c;
	checkCudaErrors(cufftPlan1d(&plan_all_echo_c2c, paras.range_num, CUFFT_C2C, paras.echo_num));
	cufftHandle plan_one_echo_c2c;
	checkCudaErrors(cufftPlan1d(&plan_one_echo_c2c, paras.range_num, CUFFT_C2C, 1));
	cufftHandle plan_one_echo_r2c;  // implicitly forward
	checkCudaErrors(cufftPlan1d(&plan_one_echo_r2c, paras.range_num, CUFFT_R2C, 1));
	cufftHandle plan_one_echo_c2r;  // implicitly inverse
	checkCudaErrors(cufftPlan1d(&plan_one_echo_c2r, paras.range_num, CUFFT_C2R, 1));

	// * Overall kernal function configuration
	dim3 block(256);  // block size
	dim3 grid((data_num + block.x - 1) / block.x);  // grid size
	dim3 grid_range((paras.range_num + block.x - 1) / block.x);

	// * GPU memory mallocation
	cuComplex* d_data = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(cuComplex) * data_num));
	checkCudaErrors(cudaMemcpy(d_data, h_data, sizeof(cuComplex) * data_num, cudaMemcpyHostToDevice));  // data (host -> device)
	thrust::device_ptr<comThr> thr_d_data = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_data));

	auto tEnd_InitGPU = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_InitGPU - tStart_InitGPU).count() << "ms\n";
	std::cout << "---* GPU Memory Initialization Over *---\n";
	std::cout << "************************************\n\n";


#ifdef DATA_WRITE_BACK
	ioOperation::dataWriteBack(std::string(DIR_PATH) + "dataW.dat", d_data, data_num);
#endif // DATA_WRITE_BACK


	/******************************
	* HPC
	******************************/
	if (ifHPC == true) {
		std::cout << "---* Starting HPC *---\n";

		// * Retrieving Velocity Data
		float* h_velocity = new float[paras.echo_num];
		std::transform(dataNOut.cbegin(), dataNOut.cend(), h_velocity, [](const std::vector<float>& v) {return v[1]; });

		float* d_velocity = nullptr;
		checkCudaErrors(cudaMalloc((void**)&d_velocity, sizeof(float) * paras.echo_num));
		checkCudaErrors(cudaMemcpy(d_velocity, h_velocity, sizeof(float) * paras.echo_num, cudaMemcpyHostToDevice));

		auto tStart_HPC = std::chrono::high_resolution_clock::now();

		// * Starting HPC
		highSpeedCompensation(d_data, paras.Fs, paras.band_width, paras.Tp, d_velocity, paras.echo_num, paras.range_num, handle);
		
		auto tEnd_HPC = std::chrono::high_resolution_clock::now();

		delete[] h_velocity;
		h_velocity = nullptr;
		checkCudaErrors(cudaFree(d_velocity));
		d_velocity = nullptr;

		std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_HPC - tStart_HPC).count() << "ms\n";
		std::cout << "---* HPC Over *---\n";
		std::cout << "************************************\n\n";
	}


#ifdef DATA_WRITE_BACK
	ioOperation::dataWriteBack(std::string(DIR_PATH) + "hpc.dat", d_data, data_num);
#endif // DATA_WRITE_BACK


	/******************
	 * HRRP
	 ******************/
	std::cout << "---* Starting Get HRRP *---\n";

	// * Generate Hamming Window
	float* hamming = nullptr;
	checkCudaErrors(cudaMalloc((void**)&hamming, sizeof(float) * paras.range_num));
	genHammingVec << <grid_range, block >> > (hamming, paras.range_num);
	checkCudaErrors(cudaDeviceSynchronize());

	// * HRRP - High Resolution Range Profile.
	cuComplex* d_hrrp = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_hrrp, sizeof(cuComplex) * data_num));

	auto tStart_HRRP = std::chrono::high_resolution_clock::now();

	// d_hrrp = fftshift(fft(hamming ,* d_data))
	getHRRP(d_hrrp, d_data, paras.echo_num, paras.range_num, hamming, plan_all_echo_c2c);

	auto tEnd_HRRP = std::chrono::high_resolution_clock::now();

	std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_HRRP - tStart_HRRP).count() << "ms\n";
	std::cout << "---* Get HRRP Over *---\n";
	std::cout << "************************************\n\n";


	/******************
	 * Range Alignment and HRRP Centering
	 ******************/
	std::cout << "---* Starting Range alignment *---\n";
	auto tStart_RA = std::chrono::high_resolution_clock::now();

	// * Range Alignment
	rangeAlignment(d_data, hamming, paras, handle, plan_one_echo_c2c, plan_one_echo_r2c, plan_one_echo_c2r);

	auto tEnd_RA_1 = std::chrono::high_resolution_clock::now();

	// * Centering HRRP
	unsigned int inter_length = 30;
	HRRPCenter(d_data, paras, inter_length, handle, plan_all_echo_c2c);

	auto tEnd_RA_2 = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption(range alignment)] " << std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_RA_1 - tStart_RA).count() << "ms\n";
	std::cout << "[Time consumption(centering HRRP)] " << std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_RA_2 - tEnd_RA_1).count() << "ms\n";
	std::cout << "---* Range alignment Over *---\n";
	std::cout << "************************************\n\n";


	/******************
	* Cut range profiles
	******************/
	std::cout << "---* Starting Cut range profiles *---\n";
	auto tStart_cut = std::chrono::high_resolution_clock::now();

	const int range_length = 512;

	cuComplex* d_data_cut = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_data_cut, sizeof(cuComplex)* range_length* paras.echo_num));

	// YSQ 改为求最大值点，确定开始距离单元
	float* range_abs = nullptr;  // 求幅度
	checkCudaErrors(cudaMalloc((void**)&range_abs, sizeof(float)* paras.range_num));
	thrust::device_ptr<float> thr_range_abs(range_abs);
	thrust::transform(thrust::device, thr_d_data, thr_d_data + paras.range_num, thr_range_abs,
		[]__host__ __device__(const thrust::complex<float>& x) { return thrust::abs(x); });
	thrust::device_ptr<float> min_ptr = thrust::max_element(thr_range_abs, thr_range_abs + paras.range_num);

	int mPos = static_cast<int>(&min_ptr[0] - &thr_range_abs[0]);
	paras.Pos = mPos;  // modify paras value

	cutRangeProfile(d_data, d_data_cut, range_length, paras);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(range_abs));
	auto tEnd_cut = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption] "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_cut - tStart_cut).count() << "ms\n";
	std::cout << "---* Cut range profiles Over *---\n";
	std::cout << "************************************\n\n";

	checkCudaErrors(cudaFree(d_data));  // use d_data_cut instead
	paras.range_num = range_length;  // modify paras value
	data_num = paras.echo_num * paras.range_num;


	/**********************
	 * Phase Compensation
	 * 多普勒跟踪 -> 距离向空变的相位补偿 -> 快速最小熵 (Doppler_Tracking -> RangeVariantPhaseComp -> Fast_Entropy)
	 **********************/
	float* h_azimuth = new float[paras.echo_num];
	float* h_pitch = new float[paras.echo_num];
	std::transform(dataNOut.cbegin(), dataNOut.cend(), h_azimuth, [](std::vector<float> v) {return v[2]; });  // make zip with range and velocity data
	std::transform(dataNOut.cbegin(), dataNOut.cend(), h_pitch, [](std::vector<float> v) {return v[3]; });

	// Doppler centriod tracing
	std::cout << "---* Starting Doppler centriod tracing *---\n";
	auto tStart_droptrace = std::chrono::high_resolution_clock::now();

	Doppler_Tracking(d_data_cut, paras);
	checkCudaErrors(cudaDeviceSynchronize());
	
	auto tEnd_droptrace = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption] "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_droptrace - tStart_droptrace).count() << "ms\n";
	std::cout << "---* Doppler centriod tracing Over *---\n";
	std::cout << "************************************\n\n";

	// range variant phase compensation
	std::cout << "---* Starting range variant phase compensation *---\n";
	auto tStart_ran = std::chrono::high_resolution_clock::now();

	RangeVariantPhaseComp(d_data_cut, paras, h_azimuth, h_pitch);

	checkCudaErrors(cudaDeviceSynchronize());
	auto tEnd_ran = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption] "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_ran - tStart_ran).count() << "ms\n";
	std::cout << "---* range variant phase compensation Over *---\n";
	std::cout << "************************************\n\n";

	// 
	Fast_Entropy(d_data_cut, paras);


	ioOperation::dataWriteBack(std::string(DIR_PATH) + "isar_image.dat", d_data_cut, data_num);


	/**********************
	* Free Allocated Memory & Destory Pointer
	**********************/
	delete[] h_azimuth;
	h_azimuth = nullptr;
	delete[] h_pitch;
	h_pitch = nullptr;
	h_data = nullptr;  // h_data cannot be deleted (pointer resides in vector)

	checkCudaErrors(cudaFree(hamming));
	hamming = nullptr;
	checkCudaErrors(cudaFree(d_hrrp));
	d_hrrp = nullptr;
	
	checkCudaErrors(cublasDestroy(handle));
	
	checkCudaErrors(cufftDestroy(plan_all_echo_c2c));
	checkCudaErrors(cufftDestroy(plan_one_echo_c2c));
	return EXIT_SUCCESS;
}