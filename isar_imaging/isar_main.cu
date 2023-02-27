#include "isar_main.cuh"

int ISAR_RD_Imaging_Main_Ku(RadarParameters& paras, const int& datastyle, const vec1D_COM_FLOAT& dataW, const vec2D_FLOAT& dataNOut, const int& optionAligment, const int& optionAPhase, const bool& ifHPC, const bool& ifMTRC) {
	/******************************
	* Init GPU Device
	******************************/
	int devId = findCudaDevice(0, static_cast<const char**>(nullptr));  // initialization GPU Device with no command line arguments
	if (devId == -1) {
		return EXIT_FAILURE;
	}

	
	/******************************
	* Load Echo Data
	******************************/
	std::cout << "---* Starting Load Echo Data *---\n";
	auto tStart_LoadingData = std::chrono::high_resolution_clock::now();

	const std::complex<float>* h_data = dataW.data();
	int data_num = paras.num_echoes * paras.num_range_bins;

	// Overall cuBlas handle
	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	//GPU memory mallocation
	cuComplex* d_data = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(cuComplex) * data_num));
	checkCudaErrors(cudaMemcpy(d_data, h_data, sizeof(cuComplex) * data_num, cudaMemcpyHostToDevice));  // data (host -> device)
	thrust::device_ptr<comThr> thr_d_data = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_data));

	auto tEnd_LoadingData = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_LoadingData - tStart_LoadingData).count() << "ms\n";
	std::cout << "---* Load Echo Data Over *---\n";
	std::cout << "************************************\n\n";


#ifdef DATA_WRITE_BACK
	ioOperation::dataWriteBack(std::string(DIR_PATH) + "dataW.dat", d_data, data_num);
#endif // DATA_WRITE_BACK


	/******************************
	* HPC
	******************************/
	if (ifHPC == true) {
		std::cout << "---* Starting HPC *---\n";
		auto tStart_HPC = std::chrono::high_resolution_clock::now();

		float* h_velocity = new float[paras.num_echoes];
		std::transform(dataNOut.cbegin(), dataNOut.cend(), h_velocity, [](const std::vector<float>& v) {return v[1]; });

		highSpeedCompensation(d_data, paras.Fs, paras.band_width, paras.Tp, h_velocity, paras.num_echoes, paras.num_range_bins, handle);
		checkCudaErrors(cudaDeviceSynchronize());

		delete[] h_velocity;
		h_velocity = nullptr;

		auto tEnd_HPC = std::chrono::high_resolution_clock::now();
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
	auto tStart_HRRP = std::chrono::high_resolution_clock::now();

	thrust::device_vector<int> fftshift_vec(data_num);
	genFFTShiftVec(fftshift_vec);

	// HRRP - High Resolution Range Profile. 
	cuComplex* d_hrrp = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_hrrp, sizeof(cuComplex) * data_num));
	getHRRP(d_hrrp, d_data, paras.num_echoes, paras.num_range_bins, fftshift_vec);  // d_hrrp = fftshift(fft(d_data))

	checkCudaErrors(cudaDeviceSynchronize());

	auto tEnd_HRRP = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_HRRP - tStart_HRRP).count() << "ms\n";
	std::cout << "---* Get HRRP Over *---\n";
	std::cout << "************************************\n\n";



	/******************
	 * Range Alignment and HRRP Centering
	 ******************/
	// Range Alignment
	std::cout << "---* Starting Range alignment *---\n";
	auto tStart_RA = std::chrono::high_resolution_clock::now();

	RangeAlignment_linej(d_data, paras, fftshift_vec, handle);

	// HRRPCenter
	unsigned int inter_length = 30;
	HRRPCenter(d_data, paras, inter_length, handle);

	checkCudaErrors(cudaDeviceSynchronize());

	auto tEnd_RA = std::chrono::high_resolution_clock::now();
	std::cout << "[Time consumption] "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_RA - tStart_RA).count() << "ms\n";
	std::cout << "---* Range alignment Over *---\n";
	std::cout << "************************************\n\n";


	/******************
	* Cut range profiles
	******************/
	std::cout << "---* Starting Cut range profiles *---\n";
	auto tStart_cut = std::chrono::high_resolution_clock::now();

	const int range_length = 512;

	cuComplex* d_data_cut = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_data_cut, sizeof(cuComplex)* range_length* paras.num_echoes));

	// YSQ 改为求最大值点，确定开始距离单元
	float* range_abs = nullptr;  // 求幅度
	checkCudaErrors(cudaMalloc((void**)&range_abs, sizeof(float)* paras.num_range_bins));
	thrust::device_ptr<float> thr_range_abs(range_abs);
	thrust::transform(thrust::device, thr_d_data, thr_d_data + paras.num_range_bins, thr_range_abs,
		[]__host__ __device__(const thrust::complex<float>& x) { return thrust::abs(x); });
	thrust::device_ptr<float> min_ptr = thrust::max_element(thr_range_abs, thr_range_abs + paras.num_range_bins);

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
	paras.num_range_bins = range_length;  // modify paras value
	data_num = paras.num_echoes * paras.num_range_bins;


	/**********************
	 * Phase Compensation
	 * 多普勒跟踪 -> 距离向空变的相位补偿 -> 快速最小熵 (Doppler_Tracking -> RangeVariantPhaseComp -> Fast_Entropy)
	 **********************/
	float* h_azimuth = new float[paras.num_echoes];
	float* h_pitch = new float[paras.num_echoes];
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


	delete[] h_azimuth;
	h_azimuth = nullptr;
	delete[] h_pitch;
	h_pitch = nullptr;
	checkCudaErrors(cublasDestroy(handle));
	h_data = nullptr;  // h_data cannot be deleted
	return EXIT_SUCCESS;
}