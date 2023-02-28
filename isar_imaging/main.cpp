#include "common.cuh"
#include "isar_main.cuh"

int main()
{
    ioOperation* io = new ioOperation(DIR_PATH, 1);

    /******************************
    * pre processing data
    ******************************/
    std::cout << "---* Starting Pre Processing *---\n";
    auto tStart_PreProcessing = std::chrono::high_resolution_clock::now();

    RadarParameters paras{};
    int frameLength = 0;
    int frameNum = 0;
    io->getSystemParasFirstFileStretch(&paras, &frameLength, &frameNum);

    vec2D_FLOAT dataN;
    vec2D_INT stretchIndex;
    std::vector<float> turnAngle;
    int pulse_num_all = 0;
    io->readKuIFDSALLNBStretch(&dataN, &stretchIndex, &turnAngle, &pulse_num_all, paras, frameLength, frameNum);

    int CQ = 1;
    int windowHead = 10 - 1;
    int windowLength = 256;
    bool nonUniformSampling = false;
    int flagDataEnd = 0;
    std::vector<int> dataWFileSn;
    vec2D_FLOAT dataNOut;
    std::vector<float> turnAngleOut;
    io->getKuDatafileSn(&flagDataEnd, &dataWFileSn, &dataNOut, &turnAngleOut, dataN, paras, turnAngle, CQ, windowHead, windowLength, nonUniformSampling);

    vec1D_COM_FLOAT dataW;
    std::vector<int> frameHeader;
    io->getKuDataStretch(&dataW, &frameHeader, stretchIndex, dataWFileSn);
    paras.echo_num = static_cast<int>(dataWFileSn.size());
    paras.range_num = static_cast<int>(dataW.size()) / paras.echo_num;

    auto tEnd_PreProcessing = std::chrono::high_resolution_clock::now();
    std::cout << "[Time consumption] "
        << std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_PreProcessing - tStart_PreProcessing).count() << "ms\n";
    std::cout << "---* Pre Processing Over *---\n";
    std::cout << "************************************\n\n";


    /******************************
    * ISAR Main
    ******************************/
    int dataStyle = 2;
    int BL = 0;
    int XW = 1;
    bool ifHPC = 1;  // High Speed Motion Compensation
    bool ifMTRC = 0;  // Migration Through Resolution Cells
    ISAR_RD_Imaging_Main_Ku(paras, dataStyle, dataW, dataNOut, BL, XW, ifHPC, ifMTRC);


    delete io;
    io = nullptr;
    return EXIT_SUCCESS;
}