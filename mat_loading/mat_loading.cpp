/*
 * MAT-file diagnose program
 *
 * See the MATLAB API Guide for compiling information.
 *
 * Calling syntax:
 *
 *   matdgns <matfile>
 *
 * It will diagnose the MAT-file named <matfile>.
 *
 * This program demonstrates the use of the following functions:
 *
 *  matClose
 *  matGetDir
 *  matGetNextVariable
 *  matGetNextVariableInfo
 *  matOpen
 *
 * Copyright 1984-2003 The MathWorks, Inc.
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <complex>
#include <vector>
#include "mat.h"


int diagnose(const char* file) {
    MATFile* pmat;
    const char** dir;
    const char* name;
    int	  ndir;
    int	  i;
    mxArray* pa;

    printf("Reading file %s...\n\n", file);

    /*
     * Open file to get directory
     */
    pmat = matOpen(file, "r");
    if (pmat == NULL) {
        printf("Error opening file %s\n", file);
        return(1);
    }

    /*
     * get directory of MAT-file
     */
    dir = (const char**)matGetDir(pmat, &ndir);
    if (dir == NULL) {
        printf("Error reading directory of file %s\n", file);
        return(1);
    }
    else {
        printf("Directory of %s:\n", file);
        for (i = 0; i < ndir; i++)
            printf("%s\n", dir[i]);
    }
    mxFree(dir);

    /* In order to use matGetNextXXX correctly, reopen file to read in headers. */
    if (matClose(pmat) != 0) {
        printf("Error closing file %s\n", file);
        return(1);
    }
    pmat = matOpen(file, "r");
    if (pmat == NULL) {
        printf("Error reopening file %s\n", file);
        return(1);
    }

    /* Get headers of all variables */
    printf("\nExamining the header for each variable:\n");
    for (i = 0; i < ndir; i++) {
        pa = matGetNextVariableInfo(pmat, &name);
        if (pa == NULL) {
            printf("Error reading in file %s\n", file);
            return(1);
        }
        /* Diagnose header pa */
        printf("According to its header, array %s has %d dimensions\n",
            name, mxGetNumberOfDimensions(pa));
        if (mxIsFromGlobalWS(pa))
            printf("  and was a global variable when saved\n");
        else
            printf("  and was a local variable when saved\n");
        mxDestroyArray(pa);
    }

    /* Reopen file to read in actual arrays. */
    if (matClose(pmat) != 0) {
        printf("Error closing file %s\n", file);
        return(1);
    }
    pmat = matOpen(file, "r");
    if (pmat == NULL) {
        printf("Error reopening file %s\n", file);
        return(1);
    }

    /* Read in each array. */
    printf("\nReading in the actual array contents:\n");
    for (i = 0; i < ndir; i++) {
        pa = matGetNextVariable(pmat, &name);
        if (pa == NULL) {
            printf("Error reading in file %s\n", file);
            return(1);
        }
        /*
         * Diagnose array pa
         */
        printf("According to its contents, array %s has %d dimensions\n",
            name, mxGetNumberOfDimensions(pa));
        if (mxIsFromGlobalWS(pa))
            printf("  and was a global variable when saved\n");
        else
            printf("  and was a local variable when saved\n");
        mxDestroyArray(pa);
    }

    if (matClose(pmat) != 0) {
        printf("Error closing file %s\n", file);
        return(1);
    }
    printf("Done\n");
    return(0);
}

int main(int argc, char** argv)
{
    //diagnose("F:\\Users\\Project\\isar_imaging\\Simulated_stretch_data_Tiangong\\Pulse_compression_data.mat");

    // Open the mat file
    const char* mat_file = "F:\\Users\\Project\\isar_imaging\\Simulated_stretch_data_Tiangong\\Pulse_compression_data_imag.mat";
    MATFile* p_mat = matOpen(mat_file, "r");
    if (p_mat == NULL) {
        std::cerr << "Error opening mat file" << std::endl;
        return 1;
    }

    // Get the array from the mat file
    mxArray* p_mx_array = matGetVariable(p_mat, "imag");
    if (p_mx_array == NULL) {
        std::cerr << "Error reading array from mat file" << std::endl;
        return 1;
    }

    // Get the size of array
    mwSize numRows = mxGetM(p_mx_array);
    mwSize numCols = mxGetN(p_mx_array);
    mwSize numElements = mxGetNumberOfElements(p_mx_array);

    std::cout << "number of rows: " << numRows << "\n";
    std::cout << "number of cols: " << numCols << "\n";
    std::cout << "number of elements: " << numElements << std::endl;

    // Allocate memory for the C++ double array
    double* myDoubleArray = new double[static_cast<int>(numElements)];

    // Copy the data from the mxArray to the C++ double array
    //memcpy(myDoubleArray, mxGetPr(p_mx_array), numElements * sizeof(double));

    double* mxData = mxGetPr(p_mx_array);
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            myDoubleArray[i * numCols + j] = mxData[i + j * numRows];
        }
    }

    // Print out the values of the C++ double array
    for (int i = 0; i < 10; i++)
    {
        //printf("%lf\n", myDoubleArray[i]);
        std::cout << myDoubleArray[i] << "\n";
    }
    std::cout << std::endl;

    // Clean up
    auto t_1 = std::chrono::high_resolution_clock::now();

    mxDestroyArray(p_mx_array);

    auto t_2 = std::chrono::high_resolution_clock::now();

    matClose(p_mat);

    auto t_3 = std::chrono::high_resolution_clock::now();

    delete[] myDoubleArray;

    auto t_4 = std::chrono::high_resolution_clock::now();


    std::cout << "mxDestroyArray: " << std::chrono::duration_cast<std::chrono::microseconds>(t_2 - t_1).count() << "us" << std::endl;
    std::cout << "matClose: " << std::chrono::duration_cast<std::chrono::microseconds>(t_3 - t_2).count() << "us" << std::endl;
    std::cout << "delete: " << std::chrono::duration_cast<std::chrono::microseconds>(t_4 - t_3).count() << "us" << std::endl;

    return 0;
}


void sim_data_extract(std::complex<float>* h_data, std::vector<std::vector<double>>* dataNOut, \
    const char* info_mat, const char* real_mat, const char* imag_mat)
{
    // * Extracting narrow band data from info_mat
    MATFile* p_mat_info = matOpen(info_mat, "r");
    if (!p_mat_info) {
        std::cerr << "Error opening mat file" << std::endl;
    }

    int echo_num = 0;
    int range_num = 0;


    // * Extracting real and imag data from real_mat and imag_mat
    // Open the mat file
    MATFile* p_mat_real = matOpen(real_mat, "r");
    MATFile* p_mat_imag = matOpen(imag_mat, "r");
    if (!p_mat_real || !p_mat_imag) {
		std::cerr << "Error opening mat file" << std::endl;
	}

    // Get variable
    mxArray* p_mx_real = matGetVariable(p_mat_real, "real");
    mxArray* p_mx_imag = matGetVariable(p_mat_imag, "imag");
    if (!p_mx_real || !p_mx_imag) {
        std::cerr << "Error reading array from mat file" << std::endl;
    }
}
