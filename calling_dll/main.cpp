#include "isar_imaging_export.h"

int main()
{

    // * Configuring imaging parameters
    int sampling_stride = 1;
    int window_head = 10 - 1;
    int window_len = 256;

    int data_style = 2;
    int option_alignment = 0;
    int option_phase = 1;
    bool if_hpc = true;
    bool if_mtrc = true;

    // * Imaging process
    isar_imaging("F:\\Users\\Project\\isar_imaging\\210425235341_047414_1383_00\\", sampling_stride, window_head, window_len, data_style, option_alignment, option_phase, if_hpc, if_mtrc);

    return 0;
}