#include "isar_imaging_export.h"

// Duplicate from common.cuh
enum DATA_TYPE
{
    DEFAULT = 0,	// 
    IFDS    = 1,	// IFDS data
    STRETCH = 2		// stretch data
};

enum POLAR_TYPE
{
    LHP = 0,	// left-hand polarization
    RHP = 1	    // right-hand polarization
};

int main()
{
    // * Imaging parameters
    std::string dir_path("C:\\Users\\Admin\\Documents\\isar_imaging_data\\180411230920_000004_1318_01");  // IFDS
    //std::string dir_path("C:\\Users\\Admin\\Documents\\isar_imaging_data\\210425235341_047414_1383_00");  // STRETCH
    int imaging_stride = 10;
    int sampling_stride = 1;
    int window_head = 0;
    int window_len = 256;
    
    int polar_type = static_cast<int>((dir_path.find("210425235341_047414_1383_00") == std::string::npos) ? POLAR_TYPE::LHP : POLAR_TYPE::RHP);
    int data_type = static_cast<int>((dir_path.find("210425235341_047414_1383_00") == std::string::npos) ? DATA_TYPE::IFDS : DATA_TYPE::STRETCH);
    int option_alignment = 0;
    int option_phase = 1;
    bool if_hpc = true;
    bool if_mtrc = true;

    // * Data declaration
    vec1D_DBL dataN;
    vec1D_FLT turnAngle;
    int frame_len = 0;
    int frame_num = 0;

    vec1D_INT dataWFileSn;
    vec1D_DBL dataNOut;
    vec1D_FLT turnAngleOut;
    vec1D_COM_FLT dataW;
    vec1D_FLT img;

    // * GPU device initialization
    gpuDevInit();

    // * Imaging process
    // Data parsing
    dataParsing(&dataN, &turnAngle, &frame_len, &frame_num, dir_path, polar_type, data_type);

    // Data initialization
    imagingMemInit(&img, &dataWFileSn, &dataNOut, &turnAngleOut, &dataW, window_len, frame_len, data_type);

    // Sequential imaging process
    for (int i = 0; i < 1; ++i) {
        // Data extracting
        dataExtracting(&dataWFileSn, &dataNOut, &turnAngleOut, &dataW, dataN, turnAngle, frame_len, frame_num, sampling_stride, window_head, window_len, data_type);

        auto t_imaging_1 = std::chrono::high_resolution_clock::now();

        // Single ISAR imaging process
        isarMainSingle(img.data(), data_type, option_alignment, option_phase, if_hpc, if_mtrc);

        auto t_imaging_2 = std::chrono::high_resolution_clock::now();
        printf("[img %2d] %3dms\n\n", i + 1, static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(t_imaging_2 - t_imaging_1).count()));

        // Write h_img data into file
        writeFileFLT(dir_path + "\\intermediate\\final_" + std::to_string(i + 1) + std::string(".dat"), img.data(), window_len * 512);
        
        window_head += imaging_stride;
    }

    // * Free allocated memory
    imagingMemDest();

    return 0;
}
