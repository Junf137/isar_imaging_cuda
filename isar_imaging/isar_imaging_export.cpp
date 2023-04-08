#include "isar_imaging_export.h"
#include "isar_main.cuh"


int isar_imaging(const std::string& dir_path, int sampling_stride, int window_head, int window_len, int data_style, int option_alignment, int option_phase, bool if_hpc, bool if_mtrc)
{
    ioOperation* io = new ioOperation(dir_path, 1);

    /******************************
    * pre-processing data
    ******************************/
    std::cout << "---* Starting Pre-Processing *---\n";
    auto t_pre_processing_1 = std::chrono::high_resolution_clock::now();

    RadarParameters paras{};
    int frame_len = 0;
    int frame_num = 0;
    io->getSystemParasFirstFileStretch(&paras, &frame_len, &frame_num);

    vec2D_DBL dataN;
    vec2D_INT stretchIndex;
    std::vector<float> turnAngle;
    int pulse_num_all = 0;
    io->readKuIFDSALLNBStretch(&dataN, &stretchIndex, &turnAngle, &pulse_num_all, paras, frame_len, frame_num);

    bool nonUniformSampling = false;
    int flagDataEnd = 0;
    std::vector<int> dataWFileSn;
    vec2D_DBL dataNOut;
    std::vector<float> turnAngleOut;
    io->getKuDatafileSn(&flagDataEnd, &dataWFileSn, &dataNOut, &turnAngleOut, dataN, paras, turnAngle, sampling_stride, window_head, window_len, nonUniformSampling);

    vec1D_COM_FLT dataW;
    std::vector<int> frameHeader;
    io->getKuDataStretch(&dataW, &frameHeader, stretchIndex, dataWFileSn);
    paras.echo_num = static_cast<int>(dataWFileSn.size());
    paras.range_num = static_cast<int>(dataW.size()) / paras.echo_num;
    paras.data_num = static_cast<int>(dataW.size());

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

    auto t_pre_processing_2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_pre_processing_2 - t_pre_processing_1).count() << "ms\n";
    std::cout << "---* Pre Processing Over *---\n";
    std::cout << "************************************\n\n";


    /******************************
    * ISAR Main
    ******************************/
    ISAR_RD_Imaging_Main_Ku(paras, data_style, dataW, dataNOut, option_alignment, option_phase, if_hpc, if_mtrc);


    delete io;
    io = nullptr;
    return EXIT_SUCCESS;
}