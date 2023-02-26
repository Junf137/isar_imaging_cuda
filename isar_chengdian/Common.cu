#include "Common.cuh"

/*******************************************
* 函数功能:      打印设备信息，
*                分别输出设备号、设备名称、计算能力。
* 输入参数:
* device_number：设备编号
*******************************************/
void GetDeviceInformation(int device_number)
{
	cudaDeviceProp device_propority;
	checkCudaErrors(cudaGetDeviceProperties(&device_propority, device_number));
	std::cout << "Device Infomation: " << std::endl;
	std::cout << "Device" << device_number << ": " << device_propority.name << std::endl;
	std::cout << "Compute Capability: " << device_propority.major << '.' << device_propority.minor << std::endl;
	checkCudaErrors(cudaSetDevice(device_number));
}


/*******************************************
* 函数功能:      利用thrust库进行fftshift，
*                对fft结果进行shift，实现
*                输入向量的前半部分和后半部分交换,
*                shift是一维的，
*                只适用于偶数长度的向量！！！
*                如果输入是二维数据矩阵，
*                数据在内存里排布合适也可直接用这个
*                函数一次性交换完。
*
* 输入参数:
* d_data：       fft结果
* data_length：  d_data长度，只有在偶数时结果才正确
* 备注：         之后发现先对数据乘以(1,-1,1,-1...)向量，再作fft
*                同样可以实现shift，效率可能更高（不涉及交换）
*                但需要先乘后fft，改动较大，还没来得及试验。
*                详请可参考：
*                https://forums.developer.nvidia.com/t/does-cuda-provide-fftshift-function-like-matlab/26102/4
*******************************************/
void fftshiftThrust(cuComplex* d_data, int data_length)
{
	comThr* thr_temp_d_data = reinterpret_cast<comThr*>(d_data);
	thrust::device_ptr<comThr>thr_d_data = thrust::device_pointer_cast(thr_temp_d_data);
	thrust::swap_ranges(thrust::device, thr_d_data, thr_d_data + data_length / 2, thr_d_data + data_length / 2);
}

/***********************************************************
 * 函数功能：参见上一个函数的备注，同样，length为偶数才正确
 * 备注：2020-06-28添加
 ***********************************************************/
void fftshiftMulWay(thrust::device_vector<int>& shift_vec, size_t length)
{
	thrust::sequence(thrust::device, shift_vec.begin(), shift_vec.end(), 0);
	thrust::transform(thrust::device, shift_vec.begin(), shift_vec.end(), shift_vec.begin(), []__host__ __device__(int x) { return 1 - 2 * (x & 1); });
}

/*******************************************
* 函数功能:      获取距离像序列，
*                对去斜数据，距离像是对时域回波作fft，
*                当数据存储格式是一个回波一个回波依次存在内存里时，
*                记为行主序；
*                当数据存储格式是距离单元依次存在内存里时，
*                记为列主序
*
* 输入参数:
* d_data：       回波数据
* echo_num：     慢时间（方位向）点数
* range_num:     快时间点数
* arrangement_rank:
*                列主序or行主序
*******************************************/
void GetHRRP(cuComplex* d_data, unsigned int echo_num, unsigned int range_num, const std::string& arrangement_rank)
{
	cufftHandle plan;
	if (arrangement_rank == "columnMajor") {
		int batch = echo_num;
		int rank = 1;
		int n[1] = { range_num };
		int inembed[] = { range_num };
		int onembed[] = { range_num };
		int istride = echo_num;
		int ostride = echo_num;
		int idist = 1;
		int odist = 1;

		checkCuFFTErrors(cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch));
	}
	else {
		int batch = echo_num;
		int rank = 1;
		int n[1] = { range_num };
		int inembed[] = { range_num };
		int onembed[] = { range_num };
		int istride = 1;
		int ostride = 1;
		int idist = range_num;
		int odist = range_num;

		checkCuFFTErrors(cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch));
	}

	checkCuFFTErrors(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
	checkCuFFTErrors(cufftDestroy(plan));
}


/*******************************************
* 函数功能:计算两个复数向量相乘
*          result_matrix = vec1 * vec2.';
*          result_matrix以列主序存储（例如，m*n大小的矩阵，内存里是一列一列依次存入）
*          vec1和vec2都是thrust的向量
* 输入参数:
* handle:        cublas库的句柄
* result_matrix：存放结果的矩阵
				（数据格式：列主序存放）
* vec1：         第一个参与计算的向量，
				 长度为m
* vec2：         第二个参与计算的向量，
				 长度为n
*******************************************/
void vectorMulvectorCublasC(cublasHandle_t handle, cuComplex* result_matrix, thrust::device_vector<comThr>& vec1, thrust::device_vector<comThr>& vec2, int m, int n)
{
	cuComplex alpha;
	alpha.x = 1.0f;
	alpha.y = 0.0f;
	// step 1: type convert
	cuComplex* d_vec1 = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(vec1.data()));
	cuComplex* d_vec2 = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(vec2.data()));

	// step 2: execute multiplication
	checkCublasErrors(cublasCgeru(handle, m, n, &alpha, d_vec1, 1, d_vec2, 1, result_matrix, m));
}


/*******************************************
* 函数功能: 同上，数据类型是实数
*******************************************/
void vectorMulvectorCublasf(cublasHandle_t handle, float* result_matrix, thrust::device_vector<float>& vec1, thrust::device_vector<float>& vec2, int m, int n)
{
	float alpha;
	alpha = 1.0;

	float* d_vec1 = reinterpret_cast<float*>(thrust::raw_pointer_cast(vec1.data()));
	float* d_vec2 = reinterpret_cast<float*>(thrust::raw_pointer_cast(vec2.data()));

	checkCublasErrors(cublasSger(handle, m, n, &alpha, d_vec1, 1, d_vec2, 1, result_matrix, m));
}

/******************************
 * 函数功能：实现 2^nextpow2(N)
 * 备注：    MATLAB里是只返回指数，这里一并求出了2的幂
 ******************************/
int nextPow2(int N)
{
	int n = 1;
	while (N >> 1) {
		n = n << 1;
		N = N >> 1;
	}
	n = n << 1;
	return n;
}

/*******************************************************************
 * 函数功能：    对距离像进行径向截取
 * 输入参数：
 * d_data:       距离像（格式：方位*距离，行主序）；
 * range_length: 截取后留下的点数,一般是2的幂；
 *               输入距离像中心两侧各截取range_length / 2点；
 * Paras:        雷达参数。
 ********************************************************************/
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

	cutRangeProfileHelper << <grid, block >> > (d_data, d_data_out, data_size, start_echo, range_length, num_ori_elements);
	//--
}


/********************************************
 * 函数功能：实现距离像的径向截取
 * 输入参数：
 * d_in：            原始距离像序列（按回波依次存储，下同）；
 * d_out：           截取后的距离像序列；
 * data_size：       截取后的距离像序列元素个数（行数*列数）；
 * offset：          每个回波截取的起始点；
 * num_elements：    截取后，每个回波的点数；
 * num_ori_elements：截取前，每个回波的点数。
 *********************************************/
__global__ void cutRangeProfileHelper(cuComplex* d_in, cuComplex* d_out, const int data_size,
	const int offset, const int num_elements, const int num_ori_elements)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= data_size)
		return;
	d_out[idx] = d_in[(idx / num_elements) * num_ori_elements + offset + idx % num_elements];
}

/*******************************************************************
* Function: get maximum value of every column
* Paras:
* c:        data set, column is not coalescing
* maxval:   vector to store maximum value
* maxidx:   vector to store maximum value's index
* row:      number of rows
* col:      number of columns
* 备注：    具体原理不懂，参考：
*           https://stackoverflow.com/questions/17698969/determining-the-least-element-and-its-position-in-each-matrix-column-with-cuda-t
********************************************************************/
using namespace thrust::placeholders;

void getMaxInColumns(thrust::device_vector<float>& c, thrust::device_vector<float>& maxval, thrust::device_vector<int>& maxidx, int row, int col)
{
	thrust::reduce_by_key(
		thrust::make_transform_iterator(
			thrust::make_counting_iterator((int)0),
			_1 / row),
		thrust::make_transform_iterator(
			thrust::make_counting_iterator((int)0),
			_1 / row) + row * col,
		thrust::make_zip_iterator(
			thrust::make_tuple(
				thrust::make_permutation_iterator(
					c.begin(),
					thrust::make_transform_iterator(
						thrust::make_counting_iterator((int)0), (_1 % row) * col + _1 / row)),
				thrust::make_transform_iterator(
					thrust::make_counting_iterator((int)0), _1 % row))),
		thrust::make_discard_iterator(),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				maxval.begin(),
				maxidx.begin())),
		thrust::equal_to<int>(),
		thrust::maximum<thrust::tuple<float, int> >()
	);
}

/*******************************************************************
* 函数功能：    给输入数添加汉明窗（应该是hamming，第一次把名称误写成hanning了）
* 输入参数：
* d_DataCut:    距离像（格式：方位*距离，行主序）；
* NumEcho:      回波数
* NumRange:     快时间采样点数
* 备注：        这个函数一开始是为最后成像前方位向加窗而写的，
*               窗长为NumEcho，通过乘以全1向量，将其扩充为NumEcho * NumRange（列主序），
*               但该函数也可以用在其他情况下的加窗操作，第二个参数为窗长即可。
********************************************************************/
void Add_HanningWindow(cuComplex* d_DataCut, int NumEcho, int NumRange)
{
	// 1. 生成hamming窗向量
	thrust::device_vector<float>Hanning(NumEcho, 0.0f);
	thrust::sequence(thrust::device, Hanning.begin(), Hanning.begin() + NumEcho / 2, 0.0f);
	thrust::sequence(thrust::device, Hanning.begin() + NumEcho / 2, Hanning.end(), float(NumEcho / 2 - 1), -1.0f);
	thrust::transform(thrust::device, Hanning.begin(), Hanning.end(), Hanning.begin(), Hanning_Window(NumEcho));

	// 2. 乘以全1向量，扩充为矩阵
	thrust::device_vector<float>Ones(NumRange, 1.0f);

	float* d_win_ones;
	checkCudaErrors(cudaMalloc((void**)&d_win_ones, sizeof(float) * NumEcho * NumRange));
	checkCudaErrors(cudaMemset(d_win_ones, 0.0, sizeof(float) * NumEcho * NumRange));
	float alpha = 1.0;

	//     type convert
	float* d_win = reinterpret_cast<float*>(thrust::raw_pointer_cast(Hanning.data()));
	float* d_ones = reinterpret_cast<float*>(thrust::raw_pointer_cast(Ones.data()));

	//     execute multiplication
	cublasHandle_t handle;
	checkCublasErrors(cublasCreate(&handle));
	// 08-05-2020改
	checkCublasErrors(cublasSger(handle, NumRange, NumEcho, &alpha, d_ones, 1, d_win, 1, d_win_ones, NumRange));

	// 3. 给数据加窗
	comThr* d_Data_temp = reinterpret_cast<comThr*>(d_DataCut);
	thrust::device_ptr<comThr> thr_d_Data = thrust::device_pointer_cast(d_Data_temp);
	thrust::device_ptr<float> thr_win_ones(d_win_ones);

	Float_Mul_Complex op_fc;
	thrust::transform(thrust::device, thr_win_ones, thr_win_ones + NumEcho * NumRange, thr_d_Data, thr_d_Data, op_fc);

	checkCublasErrors(cublasDestroy(handle));
	cudaFree(d_win_ones);
}

int ioOperation::ReadFile(std::string& path_real, std::string& path_imag, std::complex<float>* data)
{
	std::ifstream infile_real(path_real);
	std::ifstream infile_imag(path_imag);
	if (!infile_real.is_open() || !infile_imag.is_open()) {
		std::cout << "Cannot open the data file!\n";
		return EXIT_FAILURE;
	}

	int index = 0;
	float temp_real, temp_imag;
	while (infile_real >> temp_real && infile_imag >> temp_imag) {
		data[index] = { temp_real ,temp_imag };
		index++;
	}
	infile_real.close();
	infile_imag.close();
	return EXIT_SUCCESS;
}

int ioOperation::ReadData(std::string& path, std::complex<float>* data, float* vel, float* range, float* pit, float* azu, RadarParameters& para)
{
	std::ifstream mfile;
	mfile.open(path, std::ios::in | std::ios::binary);
	if (!mfile)
	{
		std::cout << "Cannot open the data file!\n";
		return EXIT_FAILURE;
	}
	uint32_t datalen = para.num_range_bins;
	uint32_t temp_head = 0, temp_last = 0;
	int temp = 0;
	float ftemp = 0;
	short temp_real = 0, temp_imag = 0;
	size_t index = 0;
	while (!mfile.eof())
	{
		mfile.read((char*)&temp_head, sizeof(temp_head));
		mfile.read((char*)&temp_last, sizeof(temp_last));
		if (temp_head == 0xFF12FF56 && temp_last == 0x00000002)
		{
			mfile.seekg(27 * 4, mfile.cur);
		}
		else
		{
			std::cout << "There are some mistakes in the data set\n";
			break;
		}
		mfile.read((char*)&ftemp, sizeof(float));
		range[index] = ftemp;
		mfile.seekg(4, mfile.cur);
		mfile.read((char*)&ftemp, sizeof(float));
		vel[index] = ftemp;
		mfile.seekg(78 * 4, mfile.cur);
		//方位角
		mfile.read((char*)&temp, sizeof(int));
		azu[index] = (float)temp * 360 / exp2f(24);
		mfile.read((char*)&temp, sizeof(int));
		pit[index] = (float)temp * 360 / exp2f(24);
		//索引数据头
		mfile.seekg(120 * 4, mfile.cur);
		for (int i = 0; i < datalen; i++)
		{
			mfile.read((char*)&temp_real, sizeof(short));
			mfile.read((char*)&temp_imag, sizeof(short));
			data[i + index * datalen] = { (float)temp_real, (float)temp_imag };
		}
		mfile.seekg(4, mfile.cur);
		index++;
		if (index >= para.num_echoes)
		{
			std::cout << "Finish sort the data\n";
			break;
		}
	}
	mfile.close();
	return EXIT_SUCCESS;
}


int ioOperation::FindPara(std::string& path, RadarParameters& para)
{
	std::ifstream filehead;
	filehead.open(path, std::ios::in | std::ios::binary);
	//Change the default format
	//filehead.unsetf(std::ios::hex);
	if (!filehead)
	{
		std::cout << "Cannot open the data file!\n";
		return EXIT_FAILURE;
	}
	//First step: 划分出一块内存，读取一段数据，数据应大于一帧，比如5MB
	//const int mLength = 1024 * 1024 * 2;
	//uint32_t* mHead = new uint32_t[mLength];
	//delete[] mHead;
	//filehead.read(mHead, mLength);
	//Experiment Read a line data from
	/*
	std::string line, temp_head, temp_last;
	std::getline(filehead,line);
	temp_head = string_to_hex(line);
	std::cout<< temp_head <<std::endl;
	*/

	uint32_t temp_head = 0, temp_last = 0;
	uint32_t temp;
	uint32_t mIndex = 0;
	while (filehead)
	{
		filehead.read((char*)&temp_head, sizeof(temp_head));
		filehead.read((char*)&temp_last, sizeof(temp_last));
		if (temp_head == 0xFF12FF56 && temp_last == 0x00000002)
		{
			//往后移动40个字节
			filehead.seekg(60, filehead.cur);
			break;
		}
		//读一个64位
		/*filehead >> temp;
		mIndex++;
		temp_head = static_cast<uint32_t>(temp & 0xffffffff00000000);
		temp_last = static_cast<uint32_t>(temp & 0x00000000ffffffff);
		if (temp_head == 0xFF12FF56 && temp_last == 0x00000002)
		{
			break;
		}*/
	}
	filehead.read((char*)&temp, sizeof(temp));
	//读取带宽
	para.band_width = temp * 1e5;
	//读取脉宽
	filehead.read((char*)&temp, sizeof(temp));
	para.Tp = temp / 1e7;
	//采样点数
	para.num_range_bins = para.Fs * para.Tp;
	filehead.close();
	return EXIT_SUCCESS;
}

//transform the string to hex format
std::string ioOperation::string_to_hex(const std::string& in)
{
	std::stringstream ss;

	ss << std::hex << std::setfill('0');
	for (size_t i = 0; in.length() > i; ++i) {
		ss << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(in[i
		]));
	}

	return ss.str();
}

int ioOperation::ReadFile(std::string& path, float* data)
{
	std::ifstream infile(path);
	if (!infile.is_open()) {
		std::cout << "Cannot open the data file!\n";
		return EXIT_FAILURE;
	}
	int index = 0;
	float temp;
	while (infile >> temp) {
		data[index] = temp;
		index++;
	}
	infile.close();
	return EXIT_SUCCESS;
}

int ioOperation::WriteFile(std::string& path_real, std::string& path_imag, std::complex<float>* data, size_t data_size)
{
	std::ofstream outfile_real(path_real);
	std::ofstream outfile_imag(path_imag);
	if (!outfile_real.is_open() || !outfile_imag.is_open()) {
		std::cout << "Cannot open the file\n" << std::endl;
		return EXIT_FAILURE;
	}

	for (int idx = 0; idx < data_size; idx++) {
		outfile_real << data[idx].real() << std::endl;
		outfile_imag << data[idx].imag() << std::endl;
	}

	outfile_real.close();
	outfile_imag.close();

	return EXIT_SUCCESS;
}

int ioOperation::WriteFile(std::string& path, float* data, size_t data_size)
{
	std::ofstream outfile(path);
	if (!outfile.is_open()) {
		std::cout << "Cannot open the file\n" << std::endl;
		return EXIT_FAILURE;
	}

	for (int idx = 0; idx < data_size; idx++) {
		outfile << data[idx] << std::endl;
	}

	outfile.close();
	return EXIT_SUCCESS;
}

int ioOperation::WriteFile(std::string& path, std::complex<float>* data, size_t data_size)
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