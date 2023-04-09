#include "isar_main.cuh"

#include "isar_imaging_export.h"

// * Global variables
std::string DIR_PATH;
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


/// <summary>
/// Init GPU Device.
/// Pick the device with highest Gflops/s. (single GPU mode)
/// </summary>
/// <returns></returns>
int gpuDevInit()
{
    int dev = findCudaDevice(0, static_cast<const char**>(nullptr));
    if (dev == -1) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


/// <summary>
/// Pre-processing data
/// </summary>
/// <param name="dataW"></param>
/// <param name="dataNOut"></param>
/// <param name="dir_path"></param>
/// <param name="sampling_stride"></param>
/// <param name="window_head"></param>
/// <param name="window_len"></param>
/// <returns></returns>
int dataParsing(vec2D_DBL* dataN, vec2D_INT* stretchIndex, std::vector<float>* turnAngle, \
    const std::string& dir_path)
{
#ifdef SEPARATE_TIMEING_
    std::cout << "---* Starting Data Parsing *---\n";
    auto t_data_parsing_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

    DIR_PATH.assign(dir_path);
    io.ioInit(dir_path, 1);

    int frame_len = 0;
    int frame_num = 0;
    io.getSystemParasFirstFileStretch(&paras, &frame_len, &frame_num);

    io.readKuIFDSALLNBStretch(dataN, stretchIndex, turnAngle, paras, frame_len, frame_num);

#ifdef SEPARATE_TIMEING_
    auto t_data_parsing_2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_data_parsing_2 - t_data_parsing_1).count() << "ms\n";
    std::cout << "---* Data Parsing Over *---\n";
    std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_

    return EXIT_SUCCESS;
}


int dataExtracting(vec1D_COM_FLT* dataW, vec2D_DBL* dataNOut, \
    const vec2D_DBL& dataN, const vec2D_INT& stretchIndex, const std::vector<float>& turnAngle, const int& sampling_stride, const int& window_head, const int& window_len)
{
#ifdef SEPARATE_TIMEING_
    std::cout << "---* Starting Data Extracting *---\n";
    auto t_data_extract_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

    bool nonUniformSampling = false;
    std::vector<int> dataWFileSn;
    std::vector<float> turnAngleOut;
    io.getKuDatafileSn(&dataWFileSn, dataNOut, &turnAngleOut, dataN, paras, turnAngle, sampling_stride, window_head, window_len, nonUniformSampling);

    std::vector<int> frameHeader;
    io.getKuDataStretch(dataW, &frameHeader, stretchIndex, dataWFileSn);
    paras.echo_num = static_cast<int>(dataWFileSn.size());
    paras.range_num = static_cast<int>(dataW->size()) / paras.echo_num;
    paras.data_num = static_cast<int>(dataW->size());

    if (paras.echo_num > MAX_THREAD_PER_BLOCK) {
        std::cout << "[main/WARN] echo_num > MAX_THREAD_PER_BLOCK: " << MAX_THREAD_PER_BLOCK << ", please double-check the data, then reconfiguring the parameters." << std::endl;
        return EXIT_FAILURE;
    }
    if (paras.range_num < RANGE_NUM_CUT) {
        std::cout << "[main/WARN] range_num < RANGE_NUM_CUT: " << RANGE_NUM_CUT << ", please double-check the data or optimize process of cutRangeProfile()." << std::endl;
        return EXIT_FAILURE;
    }
    if (paras.range_num < paras.echo_num) {
        std::cout << "[main/WARN] range_num < echo_num, please double-check the data." << std::endl;
        return EXIT_FAILURE;
    }

#ifdef SEPARATE_TIMEING_
    auto t_data_extract_2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_data_extract_2 - t_data_extract_1).count() << "ms\n";
    std::cout << "---* Data Extracting Over *---\n";
    std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_

    return EXIT_SUCCESS;
}


/// <summary>
/// GPU and CPU Memory Initialization.
/// </summary>
/// <param name="h_img"></param>
void imagingMemInit(float*& h_img)
{
#ifdef SEPARATE_TIMEING_
    std::cout << "---* Starting GPU Memory Initialization *---\n";
    auto t_init_gpu_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

    handles.handleInit(paras.echo_num, paras.range_num);
    h_img = new float[paras.echo_num * RANGE_NUM_CUT];

    checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(cuComplex) * paras.data_num));
    checkCudaErrors(cudaMalloc((void**)&d_data_cut, sizeof(cuComplex) * paras.echo_num * RANGE_NUM_CUT));
    checkCudaErrors(cudaMalloc((void**)&d_velocity, sizeof(double) * paras.echo_num));
    checkCudaErrors(cudaMalloc((void**)&d_hamming, sizeof(float) * paras.range_num));
    checkCudaErrors(cudaMalloc((void**)&d_hrrp, sizeof(cuComplex) * paras.data_num));
    checkCudaErrors(cudaMalloc((void**)&d_hamming_echoes, sizeof(float) * paras.echo_num));
    checkCudaErrors(cudaMalloc((void**)&d_img, sizeof(float) * paras.echo_num * RANGE_NUM_CUT));

#ifdef SEPARATE_TIMEING_
    auto t_init_gpu_2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_init_gpu_2 - t_init_gpu_1).count() << "ms\n";
    std::cout << "---* GPU Memory Initialization Over *---\n";
    std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_
}


/// <summary>
/// ISAR main imaging.
/// </summary>
/// <param name="h_img"></param>
/// <param name="data_style"></param>
/// <param name="dataW"></param>
/// <param name="dataNOut"></param>
/// <param name="option_alignment"></param>
/// <param name="option_phase"></param>
/// <param name="if_hpc"></param>
/// <param name="if_mtrc"></param>
void isarMainSingle(float* h_img, \
    const int& data_style, const std::complex<float>* h_data, const vec2D_DBL& dataNOut, const int& option_alignment, const int& option_phase, const bool& if_hpc, const bool& if_mtrc)
{
    ISAR_RD_Imaging_Main_Ku(h_img, d_data, d_data_cut, d_velocity, d_hamming, d_hrrp, d_hamming_echoes, d_img, paras, handles, data_style, h_data, dataNOut, option_alignment, option_phase, if_hpc, if_mtrc);
}


/// <summary>
/// Free Allocated Memory and Destroy Pointer.
/// </summary>
void imagingMemDest(float*& h_img)
{
    handles.handleDest();
    delete h_img;
    h_img = nullptr;

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
