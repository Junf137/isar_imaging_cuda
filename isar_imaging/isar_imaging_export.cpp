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
double* d_hamming;
cuDoubleComplex* d_hrrp;
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


int dataParsing(vec2D_DBL* dataN, vec1D_INT* stretchIndex, vec1D_FLT* turnAngle, int* frame_len, int* frame_num, \
    const std::string& file_path)
{
#ifdef SEPARATE_TIMEING_
    std::cout << "---* Starting Data Parsing *---\n";
    auto t_data_parsing_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

    io.ioInit(&INTERMEDIATE_DIR, file_path, 1);

    io.getSystemParas(&paras, frame_len, frame_num);

    io.readKuIFDSALLNBStretch(dataN, stretchIndex, turnAngle, paras, *frame_len, *frame_num);

#ifdef SEPARATE_TIMEING_
    auto t_data_parsing_2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_data_parsing_2 - t_data_parsing_1).count() << "ms\n";
    std::cout << "---* Data Parsing Over *---\n";
    std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_

    return EXIT_SUCCESS;
}


int dataExtracting(vec1D_INT* dataWFileSn, vec2D_DBL* dataNOut, vec1D_FLT* turnAngleOut, vec1D_COM_FLT* dataW, \
    const vec2D_DBL& dataN, const vec1D_INT& stretchIndex, const int frame_len, const vec1D_FLT& turnAngle, const int& sampling_stride, const int& window_head, const int& window_len)
{
#ifdef SEPARATE_TIMEING_
    std::cout << "---* Starting Data Extracting *---\n";
    auto t_data_extract_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

    bool uni_sampling = true;
    if (uni_sampling == true) {
        uniformSampling(dataWFileSn, dataNOut, turnAngleOut, dataN, turnAngle, sampling_stride, window_head, window_len);
    }
    else {
        nonUniformSampling();
    }

    vec1D_INT frameHeader;
    io.getKuDataStretch(dataW, &frameHeader, stretchIndex, frame_len, *dataWFileSn, window_len);

    // manually set range_num and data_num back to original values
    paras.echo_num = window_len;
    paras.range_num = (frame_len - 256) / 4;
    paras.data_num = paras.echo_num * paras.range_num;

#ifdef SEPARATE_TIMEING_
    auto t_data_extract_2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_data_extract_2 - t_data_extract_1).count() << "ms\n";
    std::cout << "---* Data Extracting Over *---\n";
    std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_

    return EXIT_SUCCESS;
}


void imagingMemInit(float** h_img, vec1D_INT* dataWFileSn, vec2D_DBL* dataNOut, vec1D_FLT* turnAngleOut, vec1D_COM_FLT* dataW, \
    const int& window_len, const int& frame_len)
{
#ifdef SEPARATE_TIMEING_
    std::cout << "---* Starting GPU Memory Initialization *---\n";
    auto t_init_gpu_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

    paras.echo_num = window_len;
    paras.range_num = (frame_len - 256) / 4;
    paras.data_num = paras.echo_num * paras.range_num;
    if (paras.echo_num > MAX_THREAD_PER_BLOCK) {
        std::cout << "[main/WARN] echo_num > MAX_THREAD_PER_BLOCK: " << MAX_THREAD_PER_BLOCK << ", please double-check the data, then reconfiguring the parameters." << std::endl;
        return;
    }
    if (paras.range_num < RANGE_NUM_CUT) {
        std::cout << "[main/WARN] range_num < RANGE_NUM_CUT: " << RANGE_NUM_CUT << ", please double-check the data or optimize process of cutRangeProfile()." << std::endl;
        return;
}
    if (paras.range_num < paras.echo_num) {
        std::cout << "[main/WARN] range_num < echo_num, please double-check the data." << std::endl;
        return;
    }

    dataWFileSn->resize(paras.echo_num);
    dataNOut->resize(paras.echo_num);
    turnAngleOut->resize(paras.echo_num);
    dataW->resize(paras.data_num);
    
    handles.handleInit(paras.echo_num, paras.range_num);
    *h_img = new float[paras.echo_num * RANGE_NUM_CUT];

    checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(cuComplex) * paras.data_num));
    checkCudaErrors(cudaMalloc((void**)&d_data_cut, sizeof(cuComplex) * paras.echo_num * RANGE_NUM_CUT));
    checkCudaErrors(cudaMalloc((void**)&d_velocity, sizeof(double) * paras.echo_num));
    checkCudaErrors(cudaMalloc((void**)&d_hamming, sizeof(double) * paras.range_num));
    checkCudaErrors(cudaMalloc((void**)&d_hrrp, sizeof(cuDoubleComplex) * paras.data_num));
    checkCudaErrors(cudaMalloc((void**)&d_hamming_echoes, sizeof(float) * paras.echo_num));
    checkCudaErrors(cudaMalloc((void**)&d_img, sizeof(float) * paras.echo_num * RANGE_NUM_CUT));

#ifdef SEPARATE_TIMEING_
    auto t_init_gpu_2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_init_gpu_2 - t_init_gpu_1).count() << "ms\n";
    std::cout << "---* GPU Memory Initialization Over *---\n";
    std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_
}


void parasInit(float** h_img, \
    const int& echo_num, const int& range_num, const long long& band_width, const long long& fc, const int& Fs, const double& Tp)
{
#ifdef SEPARATE_TIMEING_
    std::cout << "---* Starting GPU Memory Initialization *---\n";
    auto t_init_gpu_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

    paras.echo_num = echo_num;
    paras.range_num = range_num;
    paras.data_num = paras.echo_num * paras.range_num;
    if (paras.echo_num > MAX_THREAD_PER_BLOCK) {
        std::cout << "[main/WARN] echo_num > MAX_THREAD_PER_BLOCK: " << MAX_THREAD_PER_BLOCK << ", please double-check the data, then reconfiguring the parameters." << std::endl;
        return;
    }
    if (paras.range_num < RANGE_NUM_CUT) {
        std::cout << "[main/WARN] range_num < RANGE_NUM_CUT: " << RANGE_NUM_CUT << ", please double-check the data or optimize process of cutRangeProfile()." << std::endl;
        return;
    }
    if (paras.range_num < paras.echo_num) {
        std::cout << "[main/WARN] range_num < echo_num, please double-check the data." << std::endl;
        return;
    }


    handles.handleInit(paras.echo_num, paras.range_num);
    *h_img = new float[paras.echo_num * RANGE_NUM_CUT];

    paras.band_width = band_width;
    paras.fc = fc;
    paras.Fs = Fs;
    paras.Tp = Tp;

    checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(cuComplex) * paras.data_num));
    checkCudaErrors(cudaMalloc((void**)&d_data_cut, sizeof(cuComplex) * paras.echo_num * RANGE_NUM_CUT));
    checkCudaErrors(cudaMalloc((void**)&d_velocity, sizeof(double) * paras.echo_num));
    checkCudaErrors(cudaMalloc((void**)&d_hamming, sizeof(double) * paras.range_num));
    checkCudaErrors(cudaMalloc((void**)&d_hrrp, sizeof(cuDoubleComplex) * paras.data_num));
    checkCudaErrors(cudaMalloc((void**)&d_hamming_echoes, sizeof(float) * paras.echo_num));
    checkCudaErrors(cudaMalloc((void**)&d_img, sizeof(float) * paras.echo_num * RANGE_NUM_CUT));

#ifdef SEPARATE_TIMEING_
    auto t_init_gpu_2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_init_gpu_2 - t_init_gpu_1).count() << "ms\n";
    std::cout << "---* GPU Memory Initialization Over *---\n";
    std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_
}


void isarMainSingle(float* h_img, \
    const int& data_style, const std::complex<float>* h_data, const vec2D_DBL& dataNOut, const int& option_alignment, const int& option_phase, const bool& if_hpc, const bool& if_mtrc)
{
    ISAR_RD_Imaging_Main_Ku(h_img, d_data, d_data_cut, d_velocity, d_hamming, d_hrrp, d_hamming_echoes, d_img, paras, handles, data_style, h_data, dataNOut, option_alignment, option_phase, if_hpc, if_mtrc);
}


void imagingMemDest(float** h_img)
{
    handles.handleDest();
    delete *h_img;
    *h_img = nullptr;

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
