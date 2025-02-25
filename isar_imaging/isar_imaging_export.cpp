#include "isar_main.cuh"

#include "isar_imaging_export.h"

// * Global variables that are not exposed to exported API
std::string INTERMEDIATE_DIR;  // intermediate directory for saving intermediate results
ioOperation io;
RadarParameters paras{};
CUDAHandle handles;

cuComplex* d_data;
cuComplex* d_data_pp_1;
cuComplex* d_data_pp_2;
cuComplex* d_data_proc;
cuComplex* d_data_cut;
double* d_velocity;
float* d_hamming;
cuComplex* d_hrrp;
float* d_hamming_echoes;
float* d_img;

// Global variable used in separate compensation process
cuComplex* g_d_range_num_com_flt_1;
cuComplex* g_d_range_num_cut_com_flt_1;
cuComplex* g_d_echo_num_com_flt_1;
cuComplex* g_d_data_num_com_flt_1;
cuComplex* g_d_data_num_cut_com_flt_1;
cuComplex* g_d_hlf_data_num_com_flt_1;
float* g_d_range_num_flt_1;
float* g_d_range_num_cut_flt_1;
float* g_d_range_num_cut_flt_2;
float* g_d_echo_num_flt_1;
float* g_d_data_num_cut_flt_1;
float* g_d_data_num_flt_1;
float* g_d_data_num_flt_2;

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
#ifdef SEPARATE_TIMING_
    std::cout << "---* Starting Data Parsing *---\n";
    auto t_data_parsing_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMING_

    io.ioInit(&INTERMEDIATE_DIR, dir_path, static_cast<POLAR_TYPE>(polar_type), static_cast<DATA_TYPE>(data_type));

    io.getSystemParas(&paras, frame_len, frame_num);

    io.readKuIFDSAllNB(dataN, turnAngle, paras, *frame_len, *frame_num);

#ifdef SEPARATE_TIMING_
    auto t_data_parsing_2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_data_parsing_2 - t_data_parsing_1).count() << "ms\n";
    std::cout << "---* Data Parsing Over *---\n";
    std::cout << "************************************\n\n";
#endif // SEPARATE_TIMING_

    return EXIT_SUCCESS;
}


int dataExtracting(vec1D_INT* dataWFileSn, vec1D_DBL* dataNOut, vec1D_FLT* turnAngleOut, vec1D_COM_FLT* dataW, \
    const vec1D_DBL& dataN, const vec1D_FLT& turnAngle, const int& frame_len, const int& frame_num, const int& sampling_stride, const int& window_head, const int& window_len, int& imaging_stride, const int& data_type)
{
#ifdef SEPARATE_TIMING_
    std::cout << "---* Starting Data Extracting *---\n";
    auto t_data_extract_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMING_

    bool uni_sampling = true;
    if (uni_sampling == true) {
        io.uniformSampling(dataWFileSn, dataNOut, turnAngleOut, dataN, turnAngle, frame_num, sampling_stride, window_head, window_len);
    }
    else {
        io.nonUniformSampling();
    }

    cuComplex* d_data_old = nullptr;
    int overlap_len = 0;

    if (d_data == nullptr) {
        d_data = d_data_pp_1;
        d_data_old = nullptr;
        overlap_len = 0;
    }
    else if (d_data == d_data_pp_1) {
        d_data = d_data_pp_2;
        d_data_old = d_data_pp_1;
        overlap_len = paras.echo_num - imaging_stride;
    }
    else if (d_data == d_data_pp_2) {
        d_data = d_data_pp_1;
        d_data_old = d_data_pp_2;
        overlap_len = paras.echo_num - imaging_stride;
    }

    io.getSignalData(dataW->data(), d_data, d_data_old, d_data_proc, d_velocity, paras, *dataNOut, frame_len, frame_num, overlap_len, *dataWFileSn);

#ifdef SEPARATE_TIMING_
    auto t_data_extract_2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_data_extract_2 - t_data_extract_1).count() << "ms\n";
    std::cout << "---* Data Extracting Over *---\n";
    std::cout << "************************************\n\n";
#endif // SEPARATE_TIMING_

    return EXIT_SUCCESS;
}


void imagingMemInit(vec1D_FLT* img, vec1D_INT* dataWFileSn, vec1D_DBL* dataNOut, vec1D_FLT* turnAngleOut, vec1D_COM_FLT* dataW, \
    const int& window_len, const int& frame_len, const int& data_type, const bool& if_hpc, const bool& if_hrrp)
{
#ifdef SEPARATE_TIMING_
    std::cout << "---* Starting GPU Memory Initialization *---\n";
    auto t_init_gpu_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMING_

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

    img->resize(paras.echo_num * paras.range_num_cut);
    dataWFileSn->resize(paras.echo_num);
    dataNOut->resize(paras.echo_num * 4);
    turnAngleOut->resize(paras.echo_num);
    if (data_type == static_cast<int>(DATA_TYPE::STRETCH)) {
        dataW->resize(paras.data_num);
    }

    handles.handleInit(paras.echo_num, paras.range_num);

    d_data = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_data_pp_1, sizeof(cuComplex) * paras.data_num));
    checkCudaErrors(cudaMalloc((void**)&d_data_pp_2, sizeof(cuComplex) * paras.data_num));
    checkCudaErrors(cudaMalloc((void**)&d_data_proc, sizeof(cuComplex) * paras.data_num));
    checkCudaErrors(cudaMalloc((void**)&d_data_cut, sizeof(cuComplex) * paras.echo_num * paras.range_num_cut));
    if ((if_hpc == true) && (data_type == static_cast<int>(DATA_TYPE::STRETCH))) {
        checkCudaErrors(cudaMalloc((void**)&d_velocity, sizeof(double) * paras.echo_num));
    }
    checkCudaErrors(cudaMalloc((void**)&d_hamming, sizeof(float) * paras.range_num));
    if (if_hrrp == true) {
        checkCudaErrors(cudaMalloc((void**)&d_hrrp, sizeof(cuComplex) * paras.data_num));
    }
    checkCudaErrors(cudaMalloc((void**)&d_hamming_echoes, sizeof(float) * paras.echo_num));
    checkCudaErrors(cudaMalloc((void**)&d_img, sizeof(float) * paras.echo_num * paras.range_num_cut));

    // generate hamming window
    genHammingVecInit(d_hamming, paras.range_num, d_hamming_echoes, paras.echo_num);

    checkCudaErrors(cudaMalloc((void**)&g_d_range_num_flt_1, sizeof(float) * paras.range_num));
    checkCudaErrors(cudaMalloc((void**)&g_d_range_num_com_flt_1, sizeof(cuComplex) * paras.range_num));
    checkCudaErrors(cudaMalloc((void**)&g_d_data_num_com_flt_1, sizeof(cuComplex) * paras.data_num));
    checkCudaErrors(cudaMalloc((void**)&g_d_data_num_flt_1, sizeof(float) * paras.data_num));
    checkCudaErrors(cudaMalloc((void**)&g_d_data_num_flt_2, sizeof(float) * paras.data_num));
    checkCudaErrors(cudaMalloc((void**)&g_d_hlf_data_num_com_flt_1, sizeof(cuComplex) * paras.echo_num * (paras.range_num / 2 + 1)));  // Hermitian symmetry
    checkCudaErrors(cudaMalloc((void**)&g_d_echo_num_flt_1, sizeof(float) * paras.echo_num));
    checkCudaErrors(cudaMalloc((void**)&g_d_echo_num_com_flt_1, sizeof(cuComplex) * paras.echo_num));
    checkCudaErrors(cudaMalloc((void**)&g_d_data_num_cut_com_flt_1, sizeof(cuComplex) * paras.data_num_cut));
    checkCudaErrors(cudaMalloc((void**)&g_d_range_num_cut_flt_1, sizeof(float) * paras.range_num_cut));
    checkCudaErrors(cudaMalloc((void**)&g_d_data_num_cut_flt_1, sizeof(float) * paras.data_num_cut));
    checkCudaErrors(cudaMalloc((void**)&g_d_range_num_cut_flt_2, sizeof(float) * paras.range_num_cut));
    checkCudaErrors(cudaMalloc((void**)&g_d_range_num_cut_com_flt_1, sizeof(cuComplex) * paras.range_num_cut));

#ifdef SEPARATE_TIMING_
    auto t_init_gpu_2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_init_gpu_2 - t_init_gpu_1).count() << "ms\n";
    std::cout << "---* GPU Memory Initialization Over *---\n";
    std::cout << "************************************\n\n";
#endif // SEPARATE_TIMING_
}


void isarMainSingle(float* h_img, \
    const int& data_type, const int& option_alignment, const int& option_phase, const bool& if_hrrp, const bool& if_hpc, const bool& if_mtrc)
{
    ISAR_RD_Imaging_Main_Ku(h_img, d_data_proc, d_data_cut, paras, handles, static_cast<DATA_TYPE>(data_type), option_alignment, option_phase, if_hrrp, if_hpc, if_mtrc);
}


void writeFileFLT(const std::string& outFilePath, const float* data, const  size_t& data_size)
{
    ioOperation::writeFile(outFilePath, data, data_size);
}


void imagingMemDest(const int& data_type, const bool& if_hpc, const bool& if_hrrp)
{
    // free cuFFT handle
    handles.handleDest();

    // free allocated memory using cudaMalloc
    checkCudaErrors(cudaFree(d_data_pp_1));
    checkCudaErrors(cudaFree(d_data_pp_2));
    checkCudaErrors(cudaFree(d_data_proc));
    checkCudaErrors(cudaFree(d_data_cut));
    if ((if_hpc == true) && (data_type == static_cast<int>(DATA_TYPE::STRETCH))) {
        checkCudaErrors(cudaFree(d_velocity));
        d_velocity = nullptr;
    }
    checkCudaErrors(cudaFree(d_hamming));
    if (if_hrrp == true) {
        checkCudaErrors(cudaFree(d_hrrp));
        d_hrrp = nullptr;
    }
    checkCudaErrors(cudaFree(d_hamming_echoes));
    checkCudaErrors(cudaFree(d_img));
    d_data = nullptr;
    d_data_pp_1 = nullptr;
    d_data_pp_2 = nullptr;
    d_data_proc = nullptr;
    d_data_cut = nullptr;
    d_hamming = nullptr;
    d_hamming_echoes = nullptr;
    d_img = nullptr;

    // Free global variables
    checkCudaErrors(cudaFree(g_d_range_num_com_flt_1));
    checkCudaErrors(cudaFree(g_d_range_num_cut_com_flt_1));
    checkCudaErrors(cudaFree(g_d_echo_num_com_flt_1));
    checkCudaErrors(cudaFree(g_d_data_num_com_flt_1));
    checkCudaErrors(cudaFree(g_d_data_num_cut_com_flt_1));
    checkCudaErrors(cudaFree(g_d_hlf_data_num_com_flt_1));
    checkCudaErrors(cudaFree(g_d_range_num_flt_1));
    checkCudaErrors(cudaFree(g_d_range_num_cut_flt_1));
    checkCudaErrors(cudaFree(g_d_range_num_cut_flt_2));
    checkCudaErrors(cudaFree(g_d_echo_num_flt_1));
    checkCudaErrors(cudaFree(g_d_data_num_cut_flt_1));
    checkCudaErrors(cudaFree(g_d_data_num_flt_1));
    checkCudaErrors(cudaFree(g_d_data_num_flt_2));
    g_d_range_num_com_flt_1 = nullptr;
    g_d_range_num_cut_com_flt_1 = nullptr;
    g_d_echo_num_com_flt_1 = nullptr;
    g_d_data_num_com_flt_1 = nullptr;
    g_d_data_num_cut_com_flt_1 = nullptr;
    g_d_hlf_data_num_com_flt_1 = nullptr;
    g_d_range_num_flt_1 = nullptr;
    g_d_range_num_cut_flt_1 = nullptr;
    g_d_range_num_cut_flt_2 = nullptr;
    g_d_echo_num_flt_1 = nullptr;
    g_d_data_num_cut_flt_1 = nullptr;
    g_d_data_num_flt_1 = nullptr;
    g_d_data_num_flt_2 = nullptr;

}
