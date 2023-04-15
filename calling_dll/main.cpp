#include "isar_imaging_export.h"

#include <iostream>
#include <chrono>

int main()
{
    // * Imaging parameters
    std::string dir_path("F:\\Users\\Project\\isar_imaging\\210425235341_047414_1383_00\\");
    int imaging_stride = 10;
    int sampling_stride = 1;
    int window_head = 10 - 1;
    int window_len = 256;

    int data_style = 2;
    int option_alignment = 0;
    int option_phase = 1;
    bool if_hpc = true;
    bool if_mtrc = true;

    // * Data declaration
    vec2D_DBL dataN;
    vec1D_INT stretchIndex;
    vec1D_FLT turnAngle;
    int frame_len = 0;
    int frame_num = 0;

    vec1D_INT dataWFileSn;
    vec2D_DBL dataNOut;
    vec1D_FLT turnAngleOut;
    vec1D_COM_FLT dataW;
    float* h_img = nullptr;

    // * GPU device initialization
    gpuDevInit();

    // * Starting imaging for file in dir_path
    // * Data parsing
    dataParsing(&dataN, &stretchIndex, &turnAngle, &frame_len, &frame_num, dir_path);

    // * Data initialization
    imagingMemInit(&h_img, &dataWFileSn, &dataNOut, &turnAngleOut, &dataW,  window_len, frame_len);
    const std::complex<float>* h_data = dataW.data();

    // * Sequential imaging process
    for (int i = 0; i < 10; ++i) {
        int window_end = window_head + sampling_stride * window_len - 1;
        if (window_end > frame_num) {
            printf("[WARN] window_end > frame_num\n");
            break;
        }

        auto t_imaging_1 = std::chrono::high_resolution_clock::now();
        
        // Data extracting
        dataExtracting(&dataWFileSn, &dataNOut, &turnAngleOut, &dataW, dataN, stretchIndex, frame_len, turnAngle, sampling_stride, window_head, window_len);

        // Single ISAR imaging process
        isarMainSingle(h_img, data_style, h_data, dataNOut, option_alignment, option_phase, if_hpc, if_mtrc);
        
        window_head += imaging_stride;

        auto t_imaging_2 = std::chrono::high_resolution_clock::now();
        printf("[img %2d] %3dms\n\n", i + 1, static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(t_imaging_2 - t_imaging_1).count()));
    }

    // * Free allocated memory
    imagingMemDest(&h_img);

    return 0;
}