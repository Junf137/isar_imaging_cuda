#include "isar_main.cuh"

#include "isar_imaging_export.h"

// * Global variables that are not exposed to exported API
std::string INTERMEDIATE_DIR;  // intermediate directory for saving intermediate results
ioOperation io;
RadarParameters paras{};
CUDAHandle handles;

cuComplex* d_data;
cuComplex* d_data_cut;
double* d_velocity;
float* d_hamming;
cuComplex* d_hrrp;
float* d_hamming_echoes;
float* d_img;


int gpuDevInit()
{
    int dev = findCudaDevice(0, static_cast<const char**>(nullptr));
    if (dev == -1) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int dataParsing(vec1D_DBL* dataN, vec1D_FLT* turnAngle, int* frame_len, int* frame_num, \
    const std::string& dir_path, const int& polar_type, const int& data_type)
{
#ifdef SEPARATE_TIMEING_
    std::cout << "---* Starting Data Parsing *---\n";
    auto t_data_parsing_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

    io.ioInit(&INTERMEDIATE_DIR, dir_path, static_cast<POLAR_TYPE>(polar_type), static_cast<DATA_TYPE>(data_type));

    io.getSystemParas(&paras, frame_len, frame_num);

    io.readKuIFDSAllNB(dataN, turnAngle, paras, *frame_len, *frame_num);

#ifdef SEPARATE_TIMEING_
    auto t_data_parsing_2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_data_parsing_2 - t_data_parsing_1).count() << "ms\n";
    std::cout << "---* Data Parsing Over *---\n";
    std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_

    return EXIT_SUCCESS;
}


int dataExtracting(vec1D_INT* dataWFileSn, vec1D_DBL* dataNOut, vec1D_FLT* turnAngleOut, vec1D_COM_FLT* dataW, \
    const vec1D_DBL& dataN, const vec1D_FLT& turnAngle, const int& frame_len, const int& frame_num, const int& sampling_stride, const int& window_head, const int& window_len, const int& data_type)
{
#ifdef SEPARATE_TIMEING_
    std::cout << "---* Starting Data Extracting *---\n";
    auto t_data_extract_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

    bool uni_sampling = true;
    if (uni_sampling == true) {
        uniformSampling(dataWFileSn, dataNOut, turnAngleOut, dataN, turnAngle, frame_num, sampling_stride, window_head, window_len);
    }
    else {
        nonUniformSampling();
    }

    io.getSignalData(dataW, paras, *dataNOut, frame_len, frame_num, *dataWFileSn, window_len);

#ifdef SEPARATE_TIMEING_
    auto t_data_extract_2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_data_extract_2 - t_data_extract_1).count() << "ms\n";
    std::cout << "---* Data Extracting Over *---\n";
    std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_

    return EXIT_SUCCESS;
}


void imagingMemInit(vec1D_FLT* h_img, vec1D_INT* dataWFileSn, vec1D_DBL* dataNOut, vec1D_FLT* turnAngleOut, vec1D_COM_FLT* dataW, \
    const int& window_len, const int& frame_len, const int& data_type)
{
#ifdef SEPARATE_TIMEING_
    std::cout << "---* Starting GPU Memory Initialization *---\n";
    auto t_init_gpu_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

    paras.echo_num = window_len;
    switch (static_cast<DATA_TYPE>(data_type)) {
    case DATA_TYPE::IFDS:
        paras.range_num = RANGE_NUM_IFDS_PC;
        break;
    case DATA_TYPE::STRETCH:
        paras.range_num = (frame_len - 256) / 4;
        break;
    default:
        break;
    }
    paras.range_num_cut = RANGE_NUM_CUT;
    paras.data_num = paras.echo_num * paras.range_num;
    paras.data_num_cut = paras.echo_num * paras.range_num_cut;
    
    if (paras.echo_num > MAX_THREAD_PER_BLOCK) {
        std::cout << "[main/WARN] echo_num > MAX_THREAD_PER_BLOCK: " << MAX_THREAD_PER_BLOCK << ", please double-check the data, then reconfiguring the parameters." << std::endl;
        return;
    }
    if (paras.range_num < paras.range_num_cut) {
        std::cout << "[main/WARN] range_num < paras.range_num_cut: " << paras.range_num_cut << ", please double-check the data or optimize process of cutRangeProfile()." << std::endl;
        return;
}
    if (paras.range_num < paras.echo_num) {
        std::cout << "[main/WARN] range_num < echo_num, please double-check the data." << std::endl;
        return;
    }

    h_img->resize(paras.echo_num * paras.range_num_cut);
    dataWFileSn->resize(paras.echo_num);
    dataNOut->resize(paras.echo_num * 4);
    turnAngleOut->resize(paras.echo_num);
    dataW->resize(paras.data_num);
    
    handles.handleInit(paras.echo_num, paras.range_num);

    checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(cuComplex) * paras.data_num));
    checkCudaErrors(cudaMalloc((void**)&d_data_cut, sizeof(cuComplex) * paras.echo_num * paras.range_num_cut));
    checkCudaErrors(cudaMalloc((void**)&d_velocity, sizeof(double) * paras.echo_num));
    checkCudaErrors(cudaMalloc((void**)&d_hamming, sizeof(float) * paras.range_num));
    checkCudaErrors(cudaMalloc((void**)&d_hrrp, sizeof(cuComplex) * paras.data_num));
    checkCudaErrors(cudaMalloc((void**)&d_hamming_echoes, sizeof(float) * paras.echo_num));
    checkCudaErrors(cudaMalloc((void**)&d_img, sizeof(float) * paras.echo_num * paras.range_num_cut));

#ifdef SEPARATE_TIMEING_
    auto t_init_gpu_2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_init_gpu_2 - t_init_gpu_1).count() << "ms\n";
    std::cout << "---* GPU Memory Initialization Over *---\n";
    std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_
}


void isarMainSingle(float* h_img, \
    const int& data_type, const std::complex<float>* h_data, const vec1D_DBL& dataNOut, const int& option_alignment, const int& option_phase, const bool& if_hpc, const bool& if_mtrc)
{
    ISAR_RD_Imaging_Main_Ku(h_img, d_data, d_data_cut, d_velocity, d_hamming, d_hrrp, d_hamming_echoes, d_img, paras, handles, static_cast<DATA_TYPE>(data_type), h_data, dataNOut, option_alignment, option_phase, if_hpc, if_mtrc);
}


void writeFileFLT(const std::string& outFilePath, const float* data, const  size_t& data_size)
{
    ioOperation::writeFile(outFilePath, data, data_size);
}


void imagingMemDest()
{
    // free cuFFT handle
    handles.handleDest();

    // free allocated memory using cudaMalloc
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_data_cut));
    checkCudaErrors(cudaFree(d_velocity));
    checkCudaErrors(cudaFree(d_hamming));
    checkCudaErrors(cudaFree(d_hrrp));
    checkCudaErrors(cudaFree(d_hamming_echoes));
    checkCudaErrors(cudaFree(d_img));
    d_data = nullptr;
    d_data_cut = nullptr;
    d_velocity = nullptr;
    d_hamming = nullptr;
    d_hrrp = nullptr;
    d_hamming_echoes = nullptr;
    d_img = nullptr;
}


/******************
 * API for Simulation Data
 ******************/

//void parasInit(float** h_img, \
//    const int& echo_num, const int& range_num, const long long& band_width, const long long& fc, const int& Fs, const double& Tp)
//{
//#ifdef SEPARATE_TIMEING_
//    std::cout << "---* Starting GPU Memory Initialization *---\n";
//    auto t_init_gpu_1 = std::chrono::high_resolution_clock::now();
//#endif // SEPARATE_TIMEING_
//
//    paras.echo_num = echo_num;
//    paras.range_num = range_num;
//    paras.data_num = paras.echo_num * paras.range_num;
//    if (paras.echo_num > MAX_THREAD_PER_BLOCK) {
//        std::cout << "[main/WARN] echo_num > MAX_THREAD_PER_BLOCK: " << MAX_THREAD_PER_BLOCK << ", please double-check the data, then reconfiguring the parameters." << std::endl;
//        return;
//    }
//    if (paras.range_num < RANGE_NUM_CUT) {
//        std::cout << "[main/WARN] range_num < RANGE_NUM_CUT: " << RANGE_NUM_CUT << ", please double-check the data or optimize process of cutRangeProfile()." << std::endl;
//        return;
//    }
//    if (paras.range_num < paras.echo_num) {
//        std::cout << "[main/WARN] range_num < echo_num, please double-check the data." << std::endl;
//        return;
//    }
//
//
//    handles.handleInit(paras.echo_num, paras.range_num);
//    *h_img = new float[paras.echo_num * RANGE_NUM_CUT];
//
//    paras.band_width = band_width;
//    paras.fc = fc;
//    paras.Fs = Fs;
//    paras.Tp = Tp;
//
//    checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(cuComplex) * paras.data_num));
//    checkCudaErrors(cudaMalloc((void**)&d_data_cut, sizeof(cuComplex) * paras.echo_num * RANGE_NUM_CUT));
//    checkCudaErrors(cudaMalloc((void**)&d_velocity, sizeof(double) * paras.echo_num));
//    checkCudaErrors(cudaMalloc((void**)&d_hamming, sizeof(float) * paras.range_num));
//    checkCudaErrors(cudaMalloc((void**)&d_hrrp, sizeof(cuComplex) * paras.data_num));
//    checkCudaErrors(cudaMalloc((void**)&d_hamming_echoes, sizeof(float) * paras.echo_num));
//    checkCudaErrors(cudaMalloc((void**)&d_img, sizeof(float) * paras.echo_num * RANGE_NUM_CUT));
//
//#ifdef SEPARATE_TIMEING_
//    auto t_init_gpu_2 = std::chrono::high_resolution_clock::now();
//    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_init_gpu_2 - t_init_gpu_1).count() << "ms\n";
//    std::cout << "---* GPU Memory Initialization Over *---\n";
//    std::cout << "************************************\n\n";
//#endif // SEPARATE_TIMEING_
//}


//void sim_data_extract(std::complex<float>* h_data, std::vector<std::vector<double>>* dataNOut, \
//    const char* info_mat, const char* real_mat, const char* imag_mat)
//{
//    h_data = new std::complex<float>[paras.data_num];
//
//    // * Extracting real and imag data from real_mat and imag_mat
//    // Open the mat file
//    MATFile* p_mat_real = matOpen(real_mat, "r");
//    MATFile* p_mat_imag = matOpen(imag_mat, "r");
//    if (!p_mat_real || !p_mat_imag) {
//        std::cerr << "Error opening mat file" << std::endl;
//    }
//
//    // Get variable
//    mxArray* p_mx_real = matGetVariable(p_mat_real, "real");
//    mxArray* p_mx_imag = matGetVariable(p_mat_imag, "imag");
//    if (!p_mx_real || !p_mx_imag) {
//        std::cerr << "Error reading array from mat file" << std::endl;
//    }
//}
