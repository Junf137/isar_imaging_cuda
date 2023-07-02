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
    const vec1D_DBL& dataN, const vec1D_FLT& turnAngle, const int& frame_len, const int& frame_num, const int& sampling_stride, const int& window_head, const int& window_len, int& imaging_stride, const int& data_type)
{
#ifdef SEPARATE_TIMEING_
    std::cout << "---* Starting Data Extracting *---\n";
    auto t_data_extract_1 = std::chrono::high_resolution_clock::now();
#endif // SEPARATE_TIMEING_

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

#ifdef SEPARATE_TIMEING_
    auto t_data_extract_2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_data_extract_2 - t_data_extract_1).count() << "ms\n";
    std::cout << "---* Data Extracting Over *---\n";
    std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_

    return EXIT_SUCCESS;
}


void imagingMemInit(vec1D_FLT* img, vec1D_INT* dataWFileSn, vec1D_DBL* dataNOut, vec1D_FLT* turnAngleOut, vec1D_COM_FLT* dataW, \
    const int& window_len, const int& frame_len, const int& data_type, const bool& if_hpc, const bool& if_hrrp)
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

#ifdef SEPARATE_TIMEING_
    auto t_init_gpu_2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_init_gpu_2 - t_init_gpu_1).count() << "ms\n";
    std::cout << "---* GPU Memory Initialization Over *---\n";
    std::cout << "************************************\n\n";
#endif // SEPARATE_TIMEING_
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


/******************
 * API for Simulation Data
 ******************/
void dataParsingSim(int32_t* index_header, const std::string& dir_path)
{
    paras.band_width = 1000000000;
    paras.fc = 1800000000;
    paras.Fs = 1200000000;
    paras.Tp = 0.0002;

    // read index_header
    std::string indexHeaderFilePath = dir_path + "\\index_header.bin";
    std::ifstream indexFile(indexHeaderFilePath, std::ios::binary);
    if (!indexFile.is_open()) {
        std::cout << "[readIndexHeader/ERROR] Failed to open file: " << indexHeaderFilePath << std::endl;
        return;
    }
    indexFile.read(reinterpret_cast<char*>(index_header), sizeof(int32_t) * 1000);
    indexFile.close();

    INTERMEDIATE_DIR = dir_path + std::string("\\intermediate\\");
}


void imagingMemInitSim(vec1D_FLT* img, const int& window_len, const bool& if_hrrp)
{
    paras.echo_num = window_len;
    paras.range_num = 2384;
    paras.data_num = paras.echo_num * paras.range_num;

    paras.range_num_cut = RANGE_NUM_CUT;
    paras.data_num_cut = paras.echo_num * paras.range_num_cut;

    img->resize(paras.echo_num * paras.range_num_cut);

    handles.handleInit(paras.echo_num, paras.range_num);

    checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(cuComplex) * paras.data_num));
    checkCudaErrors(cudaMalloc((void**)&d_data_cut, sizeof(cuComplex) * paras.echo_num * paras.range_num_cut));
    checkCudaErrors(cudaMalloc((void**)&d_velocity, sizeof(double) * paras.echo_num));
    checkCudaErrors(cudaMalloc((void**)&d_hamming, sizeof(float) * paras.range_num));
    if (if_hrrp == true) {
        checkCudaErrors(cudaMalloc((void**)&d_hrrp, sizeof(cuComplex) * paras.data_num));
    }
    checkCudaErrors(cudaMalloc((void**)&d_hamming_echoes, sizeof(float) * paras.echo_num));
    checkCudaErrors(cudaMalloc((void**)&d_img, sizeof(float) * paras.echo_num * paras.range_num_cut));

    // generate hamming window
    genHammingVecInit(d_hamming, paras.range_num, d_hamming_echoes, paras.echo_num);
}


void dataExtractingSim(int32_t* index_header, const std::string& dir_path, const int& window_head, const int& window_len)
{
    int frame_length_16bits = 484768;
    int frame_length_32bits = frame_length_16bits / 2;
    int tk_len = 240000;

    float PRF = 1000.0f / 3.0f;
    float v0 = 182.4415637903265f;
    float a = 14.333349273443416f;

    int16_t* echo_buffer = new int16_t[frame_length_16bits];
    std::complex<float>* echo_buffer_complex = new std::complex<float>[frame_length_32bits];
    vec1D_COM_FLT echo_buffer_complex_vec(frame_length_32bits);

    // reading data from file
    std::string dataFilePath = dir_path + "\\tgt_0003.dat";
    std::ifstream dataFile(dataFilePath, std::ios::binary);
    if (!dataFile.is_open()) {
        std::cout << "[pulseSimData/ERROR] Failed to open file: " << dataFilePath << std::endl;
        return;
    }

    pulseCompressionSim pc_sim(paras, tk_len, frame_length_32bits);

    for (int i = 0; i < window_len; ++i) {
        dataFile.seekg((index_header[window_head + i] - 1) * 2 + 32, std::ifstream::beg);

        dataFile.read(reinterpret_cast<char*>(echo_buffer), sizeof(int16_t) * frame_length_16bits);

        for (int j = 0; (j + 3) < frame_length_32bits; j += 4) {
            echo_buffer_complex[j + 0] = std::complex<float>(echo_buffer[j * 2 + 0], echo_buffer[j * 2 + 4]);
            echo_buffer_complex[j + 1] = std::complex<float>(echo_buffer[j * 2 + 1], echo_buffer[j * 2 + 5]);
            echo_buffer_complex[j + 2] = std::complex<float>(echo_buffer[j * 2 + 2], echo_buffer[j * 2 + 6]);
            echo_buffer_complex[j + 3] = std::complex<float>(echo_buffer[j * 2 + 3], echo_buffer[j * 2 + 7]);
        }

        echo_buffer_complex_vec.assign(echo_buffer_complex, echo_buffer_complex + frame_length_32bits);

        // pulse compression
        pc_sim.pulseCompressionbyFFTSim(d_data + i * paras.range_num, echo_buffer_complex, v0 + a * i / PRF);
    }

    // free memory
    delete[] echo_buffer;
    delete[] echo_buffer_complex;
    echo_buffer = nullptr;
    echo_buffer_complex = nullptr;
}


void isarMainSingleSim(float* h_img, const bool& if_hrrp, const bool& if_mtrc)
{
    DATA_TYPE data_type = DATA_TYPE::IFDS;
    int option_alignment = 0;
    int option_phase = 0;
    bool if_hpc = false;

    ISAR_RD_Imaging_Main_Ku(h_img, d_data, d_data_cut, d_velocity, d_hamming, d_hrrp, d_hamming_echoes, d_img, paras, handles, data_type, option_alignment, option_phase, if_hrrp, if_hpc, if_mtrc);
}


void imagingMemDestSim(const bool& if_hrrp)
{
    // free cuFFT handle
    handles.handleDest();

    // free allocated memory using cudaMalloc
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_data_cut));
    checkCudaErrors(cudaFree(d_velocity));
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
    d_velocity = nullptr;
    d_hamming = nullptr;
    d_hamming_echoes = nullptr;
    d_img = nullptr;
}
