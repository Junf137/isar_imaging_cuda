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
    //>>> Simulation data
    // Imaging of simulated data;
    std::string dir_path("C:\\Users\\Admin\\Documents\\isar_imaging_data\\GBR");
    int total_len = 1000;
    int window_head = 0;
    int window_len = 256;
    int imaging_stride = 10;
    bool if_hrrp = false;
    bool if_mtrc = true;

    int32_t* index_header = new int32_t[total_len];
    vec1D_FLT img;

    // * GPU device initialization
    gpuDevInit();

    // * Imaging process
    // Data parsing
    dataParsingSim(index_header, dir_path);

    // Data initialization
    imagingMemInitSim(&img, window_len, if_hrrp);

    // Sequential imaging process
    for (int i = 0; i < 30; ++i) {
        auto t_imaging_1 = std::chrono::high_resolution_clock::now();

        // Data extracting
        dataExtractingSim(index_header, dir_path, window_head, window_len);

        auto t_imaging_2 = std::chrono::high_resolution_clock::now();

        // Single ISAR imaging process
        isarMainSingleSim(img.data(), if_hrrp, if_mtrc);

        auto t_imaging_3 = std::chrono::high_resolution_clock::now();
        printf("[img %2d] %3dms / %3dms\n\n", \
            i + 1, \
            static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(t_imaging_2 - t_imaging_1).count()), \
            static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(t_imaging_3 - t_imaging_2).count()));

        // Write h_img data into file
        //writeFileFLT(dir_path + "\\intermediate\\final_" + std::to_string(i + 1) + std::string(".dat"), img.data(), window_len * 512);

        window_head += imaging_stride;
    }

    // * Free allocated memory
    imagingMemDest(if_hrrp);
    delete[] index_header;
    index_header = nullptr;

    return 0;


    //// >>> Real radar data
    //// * Imaging parameters
    //std::string dir_path("C:\\Users\\Admin\\Documents\\isar_imaging_data\\180411230920_000004_1318_01");  // IFDS
    ////std::string dir_path("C:\\Users\\Admin\\Documents\\isar_imaging_data\\210425235341_047414_1383_00");  // STRETCH
    //int imaging_stride = 10;
    //int sampling_stride = 1;
    //int window_head = 0;
    //int window_len = 256;
    //
    //int polar_type = static_cast<int>((dir_path.find("210425235341_047414_1383_00") == std::string::npos) ? POLAR_TYPE::LHP : POLAR_TYPE::RHP);
    //int data_type = static_cast<int>((dir_path.find("210425235341_047414_1383_00") == std::string::npos) ? DATA_TYPE::IFDS : DATA_TYPE::STRETCH);
    //int option_alignment = 0;
    //int option_phase = 1;
    //bool if_hrrp = false;
    //bool if_hpc = true;
    //bool if_mtrc = true;

    //// * Data declaration
    //vec1D_DBL dataN;
    //vec1D_FLT turnAngle;
    //int frame_len = 0;
    //int frame_num = 0;

    //vec1D_INT dataWFileSn;
    //vec1D_DBL dataNOut;
    //vec1D_FLT turnAngleOut;
    //vec1D_COM_FLT dataW;
    //vec1D_FLT img;

    //// * GPU device initialization
    //gpuDevInit();

    //// * Imaging process
    //// Data parsing
    //dataParsing(&dataN, &turnAngle, &frame_len, &frame_num, dir_path, polar_type, data_type);

    //// Data initialization
    //imagingMemInit(&img, &dataWFileSn, &dataNOut, &turnAngleOut, &dataW, window_len, frame_len, data_type, if_hrrp);

    //// Sequential imaging process
    //for (int i = 0; i < 30; ++i) {
    //    if (data_type == static_cast<int>(DATA_TYPE::IFDS)) {
    //        window_head = (window_head == 50) ? 0 : window_head;
    //    }

    //    auto t_imaging_1 = std::chrono::high_resolution_clock::now();

    //    // Data extracting
    //    dataExtracting(&dataWFileSn, &dataNOut, &turnAngleOut, &dataW, dataN, turnAngle, frame_len, frame_num, sampling_stride, window_head, window_len, imaging_stride, data_type);

    //    auto t_imaging_2 = std::chrono::high_resolution_clock::now();

    //    // Single ISAR imaging process
    //    isarMainSingle(img.data(), data_type, option_alignment, option_phase, if_hrrp, if_hpc, if_mtrc);

    //    auto t_imaging_3 = std::chrono::high_resolution_clock::now();
    //    printf("[img %2d] %3dms / %3dms\n\n", \
    //        i + 1, \
    //        static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(t_imaging_2 - t_imaging_1).count()), \
    //        static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(t_imaging_3 - t_imaging_2).count()));

    //    // Write h_img data into file
    //    //writeFileFLT(dir_path + "\\intermediate\\final_" + std::to_string(i + 1) + std::string(".dat"), img.data(), window_len * 512);
    //
    //    window_head += imaging_stride;
    //}

    //// * Free allocated memory
    //imagingMemDest(if_hrrp);

    //return 0;
}
