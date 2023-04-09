#include "isar_imaging_export.h"

#include <iostream>
#include <chrono>

int main()
{
    vec2D_DBL dataN;
    vec2D_INT stretchIndex;
    std::vector<float> turnAngle;
    std::string dir_path("F:\\Users\\Project\\isar_imaging\\210425235341_047414_1383_00\\");

    float* h_img = nullptr;
    vec1D_COM_FLT dataW;
    const std::complex<float>* h_data = dataW.data();
    vec2D_DBL dataNOut;
    int sampling_stride = 1;
    int imaging_stride = 10;
    int window_head = 10 - 1;
    int window_len = 256;

    int data_style = 2;
    int option_alignment = 0;
    int option_phase = 1;
    bool if_hpc = true;
    bool if_mtrc = true;

    // * GPU device initialization
    gpuDevInit();

    // * Data parsing
    dataParsing(&dataN, &stretchIndex, &turnAngle, dir_path);

    for (int i = 0; i < 30; ++i) {
        auto t_imaging_1 = std::chrono::high_resolution_clock::now();

        // Data extracting
        dataExtracting(&dataW, &dataNOut, dataN, stretchIndex, turnAngle, sampling_stride, window_head, window_len);

        if (i == 0) {
            // GPU memory initialization
            imagingMemInit(h_img);
        }

        // Single ISAR imaging process
        h_data = dataW.data();
        isarMainSingle(h_img, data_style, h_data, dataNOut, option_alignment, option_phase, if_hpc, if_mtrc);

        window_head += imaging_stride;

        auto t_imaging_2 = std::chrono::high_resolution_clock::now();
        printf("[img %2d] %3dms\n\n", i, static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(t_imaging_2 - t_imaging_1).count()));
    }

    // * Free allocated memory
    imagingMemDest(h_img);


    return 0;
}