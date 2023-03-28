#include "common.cuh"
#include "isar_main.cuh"

int main()
{
    ioOperation* io = new ioOperation(DIR_PATH, 1);

    /******************************
    * pre-processing data
    ******************************/
    std::cout << "---* Starting Pre-Processing *---\n";
    auto t_pre_processing_1 = std::chrono::high_resolution_clock::now();

    RadarParameters paras{};
    int frame_len = 0;
    int frame_num = 0;
    io->getSystemParasFirstFileStretch(&paras, &frame_len, &frame_num);

    vec2D_FLOAT dataN;
    vec2D_INT stretchIndex;
    std::vector<float> turnAngle;
    int pulse_num_all = 0;
    io->readKuIFDSALLNBStretch(&dataN, &stretchIndex, &turnAngle, &pulse_num_all, paras, frame_len, frame_num);

    int sampling_stride = 1;
    int window_head = 10 - 1;
    int window_len = 256;
    bool nonUniformSampling = false;
    int flagDataEnd = 0;
    std::vector<int> dataWFileSn;
    vec2D_FLOAT dataNOut;
    std::vector<float> turnAngleOut;
    io->getKuDatafileSn(&flagDataEnd, &dataWFileSn, &dataNOut, &turnAngleOut, dataN, paras, turnAngle, sampling_stride, window_head, window_len, nonUniformSampling);

    vec1D_COM_FLOAT dataW;
    std::vector<int> frameHeader;
    io->getKuDataStretch(&dataW, &frameHeader, stretchIndex, dataWFileSn);
    paras.echo_num = static_cast<int>(dataWFileSn.size());
    paras.range_num = static_cast<int>(dataW.size()) / paras.echo_num;
    paras.data_num = static_cast<int>(dataW.size());

    auto t_pre_processing_2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] " << std::chrono::duration_cast<std::chrono::milliseconds>(t_pre_processing_2 - t_pre_processing_1).count() << "ms\n";
    std::cout << "---* Pre Processing Over *---\n";
    std::cout << "************************************\n\n";


    /******************************
    * ISAR Main
    ******************************/
    int data_style = 2;  // data style
    int option_alignment = 0;  // method for range alignment
    int option_phase = 1;  // method for phase adjustment
    bool if_hpc = true;  // High Speed Motion Compensation
    bool if_mtrc = true;  // Migration Through Resolution Cells
    ISAR_RD_Imaging_Main_Ku(paras, data_style, dataW, dataNOut, option_alignment, option_phase, if_hpc, if_mtrc);


    delete io;
    io = nullptr;
    return EXIT_SUCCESS;
}