#include "isar_imaging_export.h"

// duplicate from common.cuh
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

int writeFile(const std::string& outFilePath, const float* data, const  size_t& data_size);

int main()
{
    // * Imaging parameters
    //std::string dir_path("..\\..\\isar_imaging_data\\180411230920_000004_1318_01");  // IFDS
    std::string dir_path("..\\..\\isar_imaging_data\\210425235341_047414_1383_00");  // STRETCH
    int imaging_stride = 10;
    int sampling_stride = 1;
    int window_head = 10 - 1;
    int window_len = 256;
    
    int polar_type = static_cast<int>((dir_path.find("210425235341_047414_1383_00") == std::string::npos) ? POLAR_TYPE::LHP : POLAR_TYPE::RHP);
    int data_type = static_cast<int>((dir_path.find("210425235341_047414_1383_00") == std::string::npos) ? DATA_TYPE::IFDS : DATA_TYPE::STRETCH);
    int option_alignment = 0;
    int option_phase = 1;
    bool if_hpc = true;
    bool if_mtrc = true;

    // * Data declaration
    vec2D_DBL dataN;
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
    dataParsing(&dataN, &turnAngle, &frame_len, &frame_num, dir_path, polar_type, data_type);

    // * Data initialization
    if (data_type == DATA_TYPE::STRETCH) {
        imagingMemInit(&h_img, &dataWFileSn, &dataNOut, &turnAngleOut, &dataW, window_len, frame_len);
    }
    const std::complex<float>* h_data = dataW.data();

    // * Sequential imaging process
    for (int i = 0; i < 10; ++i) {
        int window_end = window_head + sampling_stride * window_len - 1;
        if (window_end > frame_num) {
            printf("[main/WARN] window_end > frame_num\n");
            break;
        }

        auto t_imaging_1 = std::chrono::high_resolution_clock::now();
        
        // Data extracting
        dataExtracting(&dataWFileSn, &dataNOut, &turnAngleOut, &dataW, dataN, frame_len, turnAngle, sampling_stride, window_head, window_len);

        // Single ISAR imaging process
        isarMainSingle(h_img, data_type, h_data, dataNOut, option_alignment, option_phase, if_hpc, if_mtrc);

        //writeFile(dir_path + "\\intermediate\\final_" + std::to_string(i + 1) + std::string(".dat"), h_img, window_len * 512);
        
        window_head += imaging_stride;

        auto t_imaging_2 = std::chrono::high_resolution_clock::now();
        printf("[img %2d] %3dms\n\n", i + 1, static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(t_imaging_2 - t_imaging_1).count()));
    }

    // * Free allocated memory
    imagingMemDest(&h_img);

    return 0;
}

int writeFile(const std::string& outFilePath, const float* data, const  size_t& data_size)
{
    std::ofstream ofs(outFilePath);
    if (!ofs.is_open()) {
        std::cout << "[writeFile/WARN] Cannot open the file: " << outFilePath << std::endl;
        return EXIT_FAILURE;
    }

    for (int idx = 0; idx < data_size; idx++) {
        ofs << std::fixed << std::setprecision(5) << data[idx] << "\n";
    }

    ofs.close();
    return EXIT_SUCCESS;
}