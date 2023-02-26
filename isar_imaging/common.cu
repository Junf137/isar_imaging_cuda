#include "common.cuh"


void cutRangeProfile(cuComplex* d_data, cuComplex* d_data_out, const int range_length, RadarParameters& Paras)
{
	// 首先进行类型转换:cuComplex->thrust
	comThr* thr_data_temp = reinterpret_cast<comThr*>(d_data);
	thrust::device_ptr<comThr>thr_data = thrust::device_pointer_cast(thr_data_temp);
	comThr* thr_data_out_temp = reinterpret_cast<comThr*>(d_data_out);
	thrust::device_ptr<comThr>thr_data_out = thrust::device_pointer_cast(thr_data_out_temp);

	// 由于输入数据格式，这里只需选出相应的回波
	//int start_echo = Paras.num_range_bins / 2 - range_length / 2;
	int start_echo = Paras.Pos - range_length / 2;
	if (start_echo < 0)
	{
		std::cout << "There is a problem on cutting the range/////\n" << std::endl;
		system("pause");
		exit(EXIT_FAILURE);
	}
	//int end_echo = Paras.num_range_bins / 2 + range_length / 2 - 1;
	//thrust::copy(thrust::device, thr_data + start_echo*Paras.num_echoes, thr_data + (end_echo + 1)*Paras.num_echoes,thr_data_out);

	// 08-05-2020修改，尚未验证
	//--
	const int num_ori_elements = Paras.num_range_bins;
	const int data_size = Paras.num_echoes * range_length;

	const int block_size = 128;
	const int grid_size = (data_size + block_size - 1) / block_size;

	dim3 block(block_size);
	dim3 grid(grid_size);

	cutRangeProfileHelper <<<grid, block >>> (d_data, d_data_out, data_size, start_echo, range_length, num_ori_elements);
	//--
}


void vectorMulvectorCublasC(cublasHandle_t handle, cuComplex* result_matrix, thrust::device_vector<comThr>& vec1, thrust::device_vector<comThr>& vec2, int m, int n)
{
	cuComplex alpha{};  // todo: ???
	alpha.x = 1.0f;
	alpha.y = 0.0f;
	// step 1: type convert
	cuComplex* d_vec1 = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(vec1.data()));
	cuComplex* d_vec2 = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(vec2.data()));

	// step 2: execute multiplication
	checkCudaErrors(cublasCgeru(handle, m, n, &alpha, d_vec1, 1, d_vec2, 1, result_matrix, m));
}


void vectorMulvectorCublasf(cublasHandle_t handle, float* result_matrix, thrust::device_vector<float>& vec1, thrust::device_vector<float>& vec2, int m, int n)
{
	float alpha;
	alpha = 1.0;

	float* d_vec1 = reinterpret_cast<float*>(thrust::raw_pointer_cast(vec1.data()));
	float* d_vec2 = reinterpret_cast<float*>(thrust::raw_pointer_cast(vec2.data()));

	checkCudaErrors(cublasSger(handle, m, n, &alpha, d_vec1, 1, d_vec2, 1, result_matrix, m));
}


void getMaxInColumns(thrust::device_vector<float>& c, thrust::device_vector<float>& maxval, thrust::device_vector<int>& maxidx, int row, int col)
{
	thrust::reduce_by_key(
		thrust::make_transform_iterator(
			thrust::make_counting_iterator((int)0),
			thrust::placeholders::_1 / row),
		thrust::make_transform_iterator(
			thrust::make_counting_iterator((int)0),
			thrust::placeholders::_1 / row) + row * col,
		thrust::make_zip_iterator(
			thrust::make_tuple(
				thrust::make_permutation_iterator(
					c.begin(),
					thrust::make_transform_iterator(
						thrust::make_counting_iterator((int)0), (thrust::placeholders::_1 % row) * col + thrust::placeholders::_1 / row)),
				thrust::make_transform_iterator(
					thrust::make_counting_iterator((int)0), thrust::placeholders::_1 % row))),
		thrust::make_discard_iterator(),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				maxval.begin(),
				maxidx.begin())),
		thrust::equal_to<int>(),
		thrust::maximum<thrust::tuple<float, int> >()
	);
}


__global__ void cutRangeProfileHelper(cuComplex* d_in, cuComplex* d_out, const int data_size,
	const int offset, const int num_elements, const int num_ori_elements)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= data_size)
		return;
	d_out[idx] = d_in[(idx / num_elements) * num_ori_elements + offset + idx % num_elements];
}

int nextPow2(int N) {
	int n = 1;
	while (N >> 1) {
		n = n << 1;
		N = N >> 1;
	}
	n = n << 1;
	return n;
}

__global__ void setNumInArray(int* arrays, int* index, int set_num, int num_index)
{
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= num_index)
		return;
	arrays[index[tid]] = set_num;
}

void genFFTShiftVec(thrust::device_vector<int>& fftshift_vec) {
	thrust::sequence(thrust::device, fftshift_vec.begin(), fftshift_vec.end(), 0);
	thrust::transform(thrust::device, fftshift_vec.begin(), fftshift_vec.end(), fftshift_vec.begin(), \
		[]__host__ __device__(int x) { return 1 - 2 * (x & 1); });
}


void getHRRP(cuComplex* d_data, unsigned int echo_num, unsigned int range_num)
{
	cufftHandle plan;

	checkCudaErrors(cufftPlan1d(&plan, range_num, CUFFT_C2C, echo_num));

	checkCudaErrors(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

	checkCudaErrors(cufftDestroy(plan));
}


float getTurnAngle(const float& azimuth1, const float& pitching1, const float& azimuth2, const float& pitching2) {
	std::vector<float> vec_1({ std::sin(pitching1 / 180 * PI_h), \
		std::cos(pitching1 / 180 * PI_h) * std::cos(azimuth1 / 180 * PI_h), \
		std::cos(pitching1 / 180 * PI_h) * std::sin(azimuth1 / 180 * PI_h) });

	std::vector<float> vec_2({ std::sin(pitching2 / 180 * PI_h), \
		std::cos(pitching2 / 180 * PI_h) * std::cos(azimuth2 / 180 * PI_h), \
		std::cos(pitching2 / 180 * PI_h) * std::sin(azimuth2 / 180 * PI_h) });

	float ret = std::acos(vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1] + vec_1[2] * vec_2[2]) / PI_h * 180;

	return ret;
}

int turnAngleLine(std::vector<float>* turnAngle, const std::vector<float>& azimuth, const std::vector<float>& pitching) {

	std::vector<int> idx;
	int pitchingSize = static_cast<int>(pitching.size());
	for (int i = 0; i < pitchingSize - 1; ++i) {
		if (std::abs(pitching[i + 1] - pitching[i]) > 0.2) {
			idx.push_back(i);
		}
	}

	std::vector<int> blkBeginNum;
	std::vector<int> blkEndNum;
	std::vector<int> blkLen;
	int idxSize = idx.empty() ? 1 : static_cast<int>(idx.size());

	blkBeginNum.insert(blkBeginNum.cend(), -1);
	blkBeginNum.insert(blkBeginNum.cend(), idx.begin(), idx.end());
	int blkSize = static_cast<int>(blkBeginNum.size());
	std::for_each(blkBeginNum.begin(), blkBeginNum.end(), [](int& x) {x++; });  // todo: add parallel execution policy


	blkEndNum.insert(blkEndNum.cend(), idx.begin(), idx.end());
	blkEndNum.insert(blkEndNum.cend(), pitchingSize - 1);

	blkLen.assign(blkSize, 0);
	std::transform(blkEndNum.cbegin(), blkEndNum.cend(), blkBeginNum.cbegin(), blkLen.begin(), [](const int& end, const int& begin) {return end - begin + 1; });

	turnAngle->assign(pitchingSize, 0);
	for (int blkIdx = 0; blkIdx < blkSize; ++blkIdx) {
		int N = blkLen[blkIdx];
		int stride = (N < 21) ? 1 : 20;
		for (int i = stride; i < N; i += stride) {
			int currentPulseNum = blkBeginNum[blkIdx] + i;
			float azimuth1 = azimuth[currentPulseNum - stride];
			float azimuth2 = azimuth[currentPulseNum];
			float pitching1 = pitching[currentPulseNum - stride];
			float pitching2 = pitching[currentPulseNum];
			float turnAngleSingle = getTurnAngle(azimuth1, pitching1, azimuth2, pitching2);
			turnAngle->at(currentPulseNum) = turnAngle->at(currentPulseNum - stride) + turnAngleSingle;  // 单帧夹角迭加
		}
		int turnAngleSize = static_cast<int>(turnAngle->size());
		for (int i = 0; i < turnAngleSize; ++i) {
			turnAngle->at(i) = std::abs(turnAngle->at(i));
		}
		if (N >= 21) {
			std::vector<int> x = [=]() { 
				std::vector<int> v; 
				for (int i = 0; (i + stride) <= N; i += stride) {
					v.push_back(i);
				}
				return v; 
			}();  // todo: range generate
			std::vector<float> Y = [=]() {
				std::vector<float> v;
				int xSize = static_cast<int>(x.size());
				for (int i = 0; i < xSize; ++i) {  // interpolation movement
					v.push_back(turnAngle->at(x[i]));
				}
				return v;
			}();
			std::vector<float> turnAngleInterp = [=]() {
				std::vector<float> v;
				for (int i = 0; i < N; ++i) {
					v.push_back(interpolate(x, Y, i, false));
				}
				return v;
			}();
			turnAngle->erase(turnAngle->cbegin() + blkBeginNum[blkIdx], turnAngle->cbegin() + blkEndNum[blkIdx] + 1);
			turnAngle->insert(turnAngle->cbegin() + blkBeginNum[blkIdx], turnAngleInterp.cbegin(), turnAngleInterp.cend());
		}

		if (blkIdx > 0) {
			for (int i = blkBeginNum[blkIdx]; i <= blkEndNum[blkIdx]; ++i) {
				turnAngle->at(i) += turnAngle->at(blkEndNum[blkIdx - 1]);
			}
		}
	}

	return EXIT_SUCCESS;
}


float interpolate(const std::vector<int>& xData, const std::vector<float>& yData, const int& x, const bool& extrapolate) {
	int size = static_cast<int>(xData.size());

	int i = 0;  // find left end of interval for interpolation
	if (x >= xData[size - 2]) {  // special case: beyond right end
		i = size - 2;
	} else {
		while (x > xData[i + 1]) i++;
	}
	float xL = static_cast<float>(xData[i]);
	float yL = yData[i];
	float xR = static_cast<float>(xData[i + 1]);
	float yR = yData[i + 1];  // points on either side (unless beyond ends)
	if (!extrapolate) {  // if beyond ends of array and not extrapolating
		if (x < xL) yR = yL;
		if (x > xR) yL = yR;
	}

	float dydx = (yR - yL) / (xR - xL);  // gradient

	return yL + dydx * (x - xL);  // linear interpolation
}

int uniformSamplingFun(int* flagDataEnd, std::vector<int>* dataWFileSn, vec2D_FLOAT* dataNOut, std::vector<float>* turnAngleOut, \
	const vec2D_FLOAT& dataN, const std::vector<float>& turnAngle, const int& CQ, const int& windowHead, const int& windowLength)
{
	int windowEnd = windowHead + CQ * windowLength - 1;
	if (windowEnd > turnAngle.size()) {
		// flag_data_end = 1;
		// DataW_FileSn = [];
		// TurnAngleOut = [];
		// DataNOut = [];
		// return;
		// % warndlg('滑窗超出数据范围，请重新调整滑窗位置', '注意');
	}
	
	// 	DataW_FileSn = WindowHead:CQ:WindowEnd;
	*dataWFileSn = [=]() {  // todo: range generate
		std::vector<int> v;
		for (int i = windowHead; (i + CQ) <= windowEnd + 1; ++i) {
			v.push_back(i);
		}
		return v;
	}();

	// TurnAngleOut = abs(TurnAngle(WindowHead:CQ:WindowEnd));
	turnAngleOut->assign(dataWFileSn->size(), 0);
	std::transform(dataWFileSn->cbegin(), dataWFileSn->cend(), turnAngleOut->begin(), [=](int x) {return std::abs(turnAngle[x]); });
	
	// DataNOut = DataN(WindowHead:CQ:WindowEnd, : );
	dataNOut->assign(dataWFileSn->size(), std::vector<float>(8, 0));
	std::transform(dataWFileSn->cbegin(), dataWFileSn->cend(), dataNOut->begin(), [=](int x) {return dataN[x]; });
	*flagDataEnd = 0;

	return EXIT_SUCCESS;
}

int nonUniformSamplingFun() {
	return EXIT_SUCCESS;
}


// ioOperation
int ioOperation::getFilePath(std::string* filePath, const std::string& dirPath, const int& fileType) {
	fs::directory_entry fsDirPath(dirPath);
	if (fsDirPath.is_directory() == false) {
		std::cout << "Invalid directory name!\n";
		return EXIT_FAILURE;
	}

	const std::vector<std::string> FILE_TYPE = { "00_1100.wbd" , "00_1101.wbd" };
	for (const auto& it : fs::directory_iterator{ fsDirPath }) {
		std::string fileStr = it.path().string();

		if (fileStr.substr(fileStr.length() - 11) == FILE_TYPE[fileType]) {
			*filePath = fileStr;
			std::cout << "---* " << *filePath << " *---\n\n";
			return EXIT_SUCCESS;
		}
	}

	return EXIT_FAILURE;
}

int ioOperation::getSystemParasFirstFileStretch(RadarParameters* paras, int* frameLength, int* frameNum, \
	const std::string& filePath, const int& fileType)
{
	std::ifstream ifs;
	ifs.open(filePath, std::ios_base::in | std::ios_base::binary);
	if (!ifs) {
		std::cout << "Cannot open file " << filePath << " !\n";
		return EXIT_FAILURE;
	}

	ifs.seekg(0, ifs.beg);

	uint32_t temp[36]{};
	ifs.read((char*)&temp, sizeof(temp));

	*frameLength = static_cast<int>(temp[4] * 4);  // 帧长度(包含帧头、正交解调数据，单位Byte)
	paras->fc = static_cast<long long>(temp[12] * 1e6);  // 信号载频
	paras->band_width = static_cast<long long>(temp[13] * 1e6);  // 信号带宽
	//float PRI = temp[14] / 1e6;  // 脉冲重复周期
	//float PRF = 1 / PRI;
	paras->Tp = static_cast<float>(temp[15] / 1e6);  // 发射脉宽
	paras->Fs = static_cast<int>((temp[17] % static_cast<int>(std::pow(2, 16))) * 1e6);  // 采样频率
	*frameNum = static_cast<int>(fs::file_size(fs::path(filePath))) / *frameLength;

	ifs.close();

	return EXIT_SUCCESS;
}

int ioOperation::readKuIFDSALLNBStretch(vec2D_FLOAT* dataN, vec2D_INT* stretchIndex, std::vector<float>* turnAngle, int* pulse_num_all, \
	const RadarParameters& paras, const int& frameLength, const int& frameNum, const std::string& filePath, const int& fileType)
{
	std::ifstream ifs;
	ifs.open(filePath, std::ios_base::in | std::ios_base::binary);
	if (!ifs) {
		std::cout << "Cannot open file " << filePath << " !\n";
		return EXIT_FAILURE;
	}

	dataN->assign(frameNum, std::vector<float>(8, 0));
	stretchIndex->assign(frameNum, std::vector<int>(2, 0));

	std::vector<float> azimuthVec(frameNum, 0);
	std::vector<float> pitchingVec(frameNum, 0);

	float timeYear = 0;  // time info only need read once
	float timeMonth = 0;
	float timeDay = 0;
	for (int i = 0; i < frameNum; i++) {
		ifs.seekg(i * frameLength + 40, ifs.beg);

		stretchIndex->at(i) = std::vector<int>({ i * frameLength + 256, frameLength });

		uint64_t sysTime = 0;
		ifs.read((char*)&sysTime, sizeof(uint64_t));

		uint32_t headerData[11]{};
		ifs.read((char*)&headerData, sizeof(headerData));

		float range = static_cast<float>(headerData[7]) * 0.1f;  // unit: m
		float velocity = static_cast<float>(headerData[8]);  // unit: m/s
		float azimuth = static_cast<float>(headerData[9]);
		float pitching = static_cast<float>(headerData[10]);

		/*if (velocity > std::pow(2, 31)) {
			velocity = (velocity - std::pow(2, 32)) * 0.1;
		}
		else {
			velocity = velocity * 0.1;
		}*/
		velocity = (velocity - (velocity > static_cast<float>(std::pow(2, 31)) ? static_cast<float>(std::pow(2, 32)) : 0)) * 0.1f;


		/*if (azimuth > std::pow(2, 31)) {
			azimuth = (azimuth - std::pow(2, 32)) * (360 / std::pow(2, 24));
		}
		else {
			azimuth = azimuth * (360 / std::pow(2, 24));
		}*/
		/*if (azimuth < 0) {
			azimuth = azimuth + 360;
		}*/
		azimuth = (azimuth - (azimuth > static_cast<float>(std::pow(2, 31)) ? static_cast<float>(std::pow(2, 32)) : 0)) * (360 / static_cast<float>(std::pow(2, 24)));
		azimuth += (azimuth < 0 ? 360 : 0);

		/*if (pitching > std::pow(2, 31)) {
			pitching = (pitching - std::pow(2, 32)) * (360 / std::pow(2, 24));
		}
		else {
			pitching = pitching * (360 / std::pow(2, 24));
		}*/
		/*if (pitching < 0) {
			pitching = pitching + 360;
		}*/
		pitching = (pitching - (pitching > static_cast<float>(std::pow(2, 31)) ? static_cast<float>(std::pow(2, 32)) : 0)) * (360 / static_cast<float>(std::pow(2, 24)));
		pitching += (pitching < 0 ? 360 : 0);

		ifs.seekg(i * frameLength + 32, ifs.beg);

		if (i == 0) {
			ifs.read((char*)&timeYear, sizeof(uint16_t));
			ifs.read((char*)&timeMonth, sizeof(uint8_t));
			ifs.read((char*)&timeDay, sizeof(uint8_t));
		}
		dataN->at(i) = std::vector<float>({ range, velocity, azimuth, pitching, static_cast<float>(sysTime), timeYear, timeMonth, timeDay });
		azimuthVec[i] = azimuth;
		pitchingVec[i] = pitching;
	}

	turnAngleLine(turnAngle, azimuthVec, pitchingVec);
	*pulse_num_all = static_cast<int>(turnAngle->size());

	ifs.close();

	return EXIT_SUCCESS;
}

int ioOperation::getKuDatafileSn(int* flagDataEnd, std::vector<int>* dataWFileSn, vec2D_FLOAT* dataNOut, std::vector<float>* turnAngleOut, \
	const vec2D_FLOAT& dataN, const RadarParameters& paras, const std::vector<float>& turnAngle, const int& CQ, const int& windowHead, const int& windowLength, const bool& nonUniformSampling)
{

	if (nonUniformSampling == true) {
		nonUniformSamplingFun();
	}
	else {
		uniformSamplingFun(flagDataEnd, dataWFileSn, dataNOut, turnAngleOut, dataN, turnAngle, CQ, windowHead, windowLength);
	}

	return EXIT_SUCCESS;
}

int ioOperation::getKuDataStretch(vec1D_COM_FLOAT* dataW, std::vector<int>* frameHeader, \
	const std::string& filePath, const vec2D_INT& stretchIndex, const std::vector<int>& dataWFileSn)
{
	std::ifstream ifs;
	ifs.open(filePath, std::ios_base::in | std::ios_base::binary);
	if (!ifs) {
		std::cout << "Cannot open file " << filePath << " !\n";
		return EXIT_FAILURE;
	}
	
	int dataWFileSnSize = static_cast<int>(dataWFileSn.size());  // row of dataW
	for (int i = 0; i < dataWFileSnSize; ++i) {
		//fseek(fid1, StretchIndex(DataW_FileSn(i), 1), 'bof');
		ifs.seekg(stretchIndex[dataWFileSn[i]][0], ifs.beg);

		//DataAD = fread(fid1, (StretchIndex(DataW_FileSn(i), 2) - 256) / 2, 'int16');
		int dataADTempSize = (stretchIndex[dataWFileSn[i]][1] - 256) / 2;  // todo: frameLength???
		int16_t* dataADTemp = new int16_t[dataADTempSize];
		ifs.read((char*)dataADTemp, dataADTempSize * sizeof(int16_t));


		if (i == 0) {
			dataW->resize(dataWFileSnSize * (dataADTempSize / 2));
		}

		//data_AD = DataAD(1:2 : end) + 1i * DataAD(2:2 : end);
		//DataW(i, :) = data_AD.';
		for (int j = 0; (j + 1) < dataADTempSize; j += 2) {
			dataW->at(i * (dataADTempSize / 2) + (j / 2)) = std::complex<float>(static_cast<float>(dataADTemp[j]), static_cast<float>(dataADTemp[j + 1]));
		}

		delete[] dataADTemp;
		dataADTemp = nullptr;
	}

	/*
	fseek(fid1, StretchIndex(DataW_FileSn(1), 1) - 256, 'bof');
	DataRead = fread(fid1, 108, 'uint8');
	FrameHeader = [DataRead(1:12, 1); DataRead(101:104, 1); DataRead(77:92, 1); DataRead(97:100, 1); DataRead(33:38, 1); DataRead(31, 1); DataRead(105:108, 1); DataRead(61:64, 1); ];
	*/
	ifs.seekg(stretchIndex[dataWFileSn[0]][0] - 256, ifs.beg);

	uint8_t frameHeaderTemp[108]{};
	ifs.read((char*)&frameHeaderTemp, sizeof(frameHeaderTemp));

	frameHeader->insert(frameHeader->cend(), frameHeaderTemp + 0, frameHeaderTemp + 12);
	frameHeader->insert(frameHeader->cend(), frameHeaderTemp + 100, frameHeaderTemp + 104);
	frameHeader->insert(frameHeader->cend(), frameHeaderTemp + 76, frameHeaderTemp + 92);
	frameHeader->insert(frameHeader->cend(), frameHeaderTemp + 96, frameHeaderTemp + 100);
	frameHeader->insert(frameHeader->cend(), frameHeaderTemp + 32, frameHeaderTemp + 38);
	frameHeader->insert(frameHeader->cend(), frameHeaderTemp + 30, frameHeaderTemp + 31);
	frameHeader->insert(frameHeader->cend(), frameHeaderTemp + 104, frameHeaderTemp + 108);
	frameHeader->insert(frameHeader->cend(), frameHeaderTemp + 60, frameHeaderTemp + 64);

	return EXIT_SUCCESS;
}


int ioOperation::WriteFile(const std::string& path, const std::complex<float>* data, const  size_t& data_size)
{
	std::ofstream outfile(path);
	if (!outfile.is_open()) {
		std::cout << "Cannot open the file\n" << std::endl;
		return EXIT_FAILURE;
	}

	for (int idx = 0; idx < data_size; idx++) {
		outfile << data[idx].real() << "\n" << data[idx].imag() << std::endl;
	}

	outfile.close();
	return EXIT_SUCCESS;
}
