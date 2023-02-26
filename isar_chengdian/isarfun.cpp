#include "isarfun.h"

void Hamming(int winLen, float* winVec)
{
    const float arg = (2.0 * PI_h) / (winLen - 1);
    for (int i = 0; i < winLen; i++)
    {
        float coef_val = 0.54 - (0.46 * cos(arg * i));
        winVec[i] = coef_val;
    }
}

MKL_Complex8 mSum(MKL_Complex8* pAdr, int mLen)
{
    MKL_Complex8 fSum = { 0, 0 };
    MKL_Complex8* pTemp = pAdr;
    for (int index = 0; index < mLen; index++)
    {
        fSum.real += pTemp->real;
        fSum.imag += pTemp->imag;
        pTemp++;
    }
    return fSum;
    pTemp = nullptr;
}

float mSumf(float* pAdr, int mLen)
{
    float fsum = 0;
    for (int i = 0; i < mLen; i++)
    {
        fsum += *pAdr;
        pAdr++;
    }
    return fsum;
}


void MiniEntropyComp(MKL_Complex8* indta, MKL_Complex8* outdta, RadarParameters& mPara)
{
    //转置
    MKL_Complex8 alpha;
    alpha.real = 1.;
    alpha.imag = 0.;
    size_t mRow = mPara.num_echoes;
    size_t mCol = mPara.num_range_bins;

    //需要迭代的相位值
    MKL_Complex8* fFail = (MKL_Complex8*)MKL_malloc(mRow * sizeof(MKL_Complex8), 64);
    //迭代的相位值的拷贝
    MKL_Complex8* fFailCopy = (MKL_Complex8*)MKL_calloc(mRow, sizeof(MKL_Complex8), 64);
    //多普勒跟踪法进行预聚焦
    MKL_Complex8* vTmpdta = (MKL_Complex8*)MKL_malloc(mRow * mCol * sizeof(MKL_Complex8), 64);
    //image1 = fft(tmpData);对列求FFT
    //转置后数据存储的区域
    MKL_Complex8* vTmptrans = (MKL_Complex8*)MKL_malloc(mRow * mCol * sizeof(MKL_Complex8), 64);
    //第一次FFT之后数据
    MKL_Complex8* vTmpFFT = (MKL_Complex8*)MKL_malloc(mRow * mCol * sizeof(MKL_Complex8), 64);

    mkl_comatcopy('R'    /* row-major ordering */,
        'T'    /* matrix will be transposed */,
        mRow     /* rows */,
        mCol     /* cols */,
        alpha /* scales the input matrix */,
        indta   /* source matrix */,
        mCol     /* src_stride */,
        vTmptrans   /* destination matrix */,
        mRow     /* dst_stride */);

    //求Max_Value
    float* vMax = (float*)MKL_malloc(mCol * sizeof(float), 64);
    int* vMaxIdx = (int*)MKL_malloc(mCol * sizeof(int), 64);

    //求Abs
    float* vTmptransAbs = (float*)MKL_malloc(mRow * mCol * sizeof(float), 64);
    //Abs
    float* vImgAbs = (float*)MKL_malloc(mRow * mCol * sizeof(float), 64);
    //二次方
    float* vImgSqr = (float*)MKL_malloc(mRow * mCol * sizeof(float), 64);
    vcAbs(mRow * mCol, vTmptrans, vTmptransAbs);
    //移动指针
    float* pMov = nullptr;
    //指针指向最大值
    float* pMax = nullptr;
    //求sum 
    float maxSum = 0;
    int mMaxIdx = 0;
    int tgtIdxNum = 0;
    for (int i = 0; i < mCol; i++)
    {
        pMov = vTmptransAbs + i * mRow;
        //利用STL开始求最大值
        pMax = std::max_element(pMov, pMov + mRow);
        vMax[i] = *pMax;
        maxSum += *pMax;
        //相对于原始位置的位置
        mMaxIdx = std::distance(vTmptransAbs, pMax);
        vMaxIdx[i] = mMaxIdx;
    }
    //求均值
    float maxMean = 1.48 * maxSum / mCol;
    //选出大于门限的Index
    for (int i = 0; i < mCol; i++)
    {
        if (vMax[i] > maxMean)
        {
            tgtIdxNum++;
        }
    }
    //首先进行多普勒预聚焦
    //DopTrace(indta, vTmpdta, mPara);
    memcpy(vTmpdta, indta, mCol * mRow * 8);
    //转置
    mkl_comatcopy('R'    /* row-major ordering */,
        'T'    /* matrix will be transposed */,
        mRow     /* rows */,
        mCol     /* cols */,
        alpha /* scales the input matrix */,
        vTmpdta   /* source matrix */,
        mCol     /* src_stride */,
        vTmptrans   /* destination matrix */,
        mRow     /* dst_stride */);

    //准备FFT
    //FFT Forward
    DFTI_DESCRIPTOR_HANDLE fHandle = NULL;
    DftiCreateDescriptor(&fHandle, DFTI_SINGLE, DFTI_COMPLEX, 1, mRow);
    DftiSetValue(fHandle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    //多次FFT
    DftiSetValue(fHandle, DFTI_NUMBER_OF_TRANSFORMS, mCol);
    //每次FFT距离
    DftiSetValue(fHandle, DFTI_INPUT_DISTANCE, mRow);
    DftiSetValue(fHandle, DFTI_OUTPUT_DISTANCE, mRow);
    //FFT Scale
    DftiSetValue(fHandle, DFTI_FORWARD_SCALE, 1.0);
    DftiSetValue(fHandle, DFTI_BACKWARD_SCALE, 1.0 / mRow);

    DftiCommitDescriptor(fHandle);
    //正向FFT
    DftiComputeForward(fHandle, vTmptrans, vTmpFFT);
    /****************
    % 方法是先找出最强的100个单元，然后在其中找出50个熵最小的
    num_unit1 = tgt_num;
    num_unit2 = num_unit1 / 2;
    sqr_image = (abs(image2)). ^ 2;
    sum_image_vector = sum(sqr_image);
    ****************/
    vcAbs(mRow * mCol, vTmpFFT, vImgAbs);
    vsSqr(mRow * mCol, vImgAbs, vImgSqr);
    //TBB并行求和
    /*
    auto mSum = tbb::parallel_reduce(tbb::blocked_range<int>(0,mRow*mCol),0.0,
        [&](tbb::blocked_range<int> r, float sumup)
        {
            for (int i = r.begin(); i < r.end(); ++i)
            {
                sumup += vImgSqr[i];
            }
        }, std::plus<float>());
    */
    float* vSum = (float*)MKL_malloc(mCol * sizeof(float), 64);
    float* vSumCopy = (float*)MKL_malloc(mCol * sizeof(float), 64);

    for (int idx = 0; idx < mCol; idx++)
    {
        pMov = vImgSqr + idx * mRow;
        vSum[idx] = mSumf(pMov, mRow);
    }
    //考虑tgtIdxNum = 0 情况
    if (tgtIdxNum <= 0)
    {
        tgtIdxNum = 128;
    }
    int mNumunit = tgtIdxNum;
    int* vSeIdx = (int*)MKL_malloc(mNumunit * sizeof(int), 64);

    int mNumunits = tgtIdxNum / 2;
    //拷贝到备份区域
    memcpy(vSumCopy, vSum, mCol * sizeof(float));
    //依次找最大
    for (int i = 0; i < mNumunit; i++)
    {
        //利用STL开始求最大值
        pMax = std::max_element(vSumCopy, vSumCopy + mCol);
        //相对于原始位置的位置
        mMaxIdx = std::distance(vSumCopy, pMax);
        vSeIdx[i] = mMaxIdx;
        //置零
        *pMax = 0;
    }
    //然后再找出50个熵最小的单元
    int mBin = 0;
    //存放归一化的图像幅值
    float* vNormal = (float*)MKL_malloc(mRow * sizeof(float), 64);
    float* vNormalLg = (float*)MKL_malloc(mRow * sizeof(float), 64);
    float* vEntroBase = (float*)MKL_malloc(mRow * sizeof(float), 64);
    //计算的Entropy以及存放Entropy的地址
    float* vEntropy = (float*)MKL_malloc(mNumunit * sizeof(float), 64);
    float fEntropy = 0;
    for (int i = 0; i < mNumunit; i++)
    {
        //先取出标号
        mBin = vSeIdx[i];
        //将指针指向对应距离单元
        pMov = vImgSqr + mBin * mRow;
        memcpy(vNormal, pMov, mRow * sizeof(float));
        cblas_sscal(mRow, 1 / vSum[mBin], vNormal, 1);
        //calculate the entropy
        vsLn(mRow, vNormal, vNormalLg);
        vsMul(mRow, vNormalLg, vNormal, vEntroBase);
        fEntropy = mSumf(vEntroBase, mRow);
        vEntropy[i] = -fEntropy;
    }
    int* vSeIdxB = (int*)MKL_malloc(mNumunits * sizeof(int), 64);
    float* pMin = nullptr;
    int vIdx = 0;
    //进一步选出最小的熵值的距离单元
    for (int i = 0; i < mNumunits; i++)
    {
        pMin = std::min_element(vEntropy, vEntropy + mNumunit);
        vIdx = std::distance(vEntropy, pMin);
        vSeIdxB[i] = vSeIdx[vIdx];
        *pMin = 200;
    }
    //提取对应距离单元
    std::sort(vSeIdxB, vSeIdxB + mNumunits);
    MKL_Complex8* vSelDta = (MKL_Complex8*)MKL_malloc(mNumunits * mRow * sizeof(MKL_Complex8), 64);
    for (int i = 0; i < mNumunits; i++)
    {
        memcpy(vSelDta + i * mRow, vTmptrans + vSeIdxB[i] * mRow, mRow * sizeof(MKL_Complex8));
    }
    /************
    %% 先用多普勒中心跟踪法求出补偿相位作为迭代的初始值
        for i = 2:M
            xw1 = data(i, :).*conj(data(i - 1, :));
    sum1 = sum(xw1);
    fai_l(i) = fai_l(i - 1) + angle(sum1);
    end
    fai_l = exp(-j * fai_l);
    ************/
    //首先转置，快时间慢时间到慢时间快时间
    MKL_Complex8* vSelTrans = (MKL_Complex8*)MKL_malloc(mNumunits * mRow * sizeof(MKL_Complex8), 64);
    MKL_Complex8* vXwl = (MKL_Complex8*)MKL_malloc(mNumunits * sizeof(MKL_Complex8), 64);
    MKL_Complex8 xwSum = { 0,0 };
    float fFSum = 0;
    float fAngle = 0;
    //转置
    mkl_comatcopy('R'    /* row-major ordering */,
        'T'    /* matrix will be transposed */,
        mNumunits     /* rows */,
        mRow     /* cols */,
        alpha /* scales the input matrix */,
        vSelDta   /* source matrix */,
        mRow     /* src_stride */,
        vSelTrans   /* destination matrix */,
        mNumunits     /* dst_stride */);
    fFail[0].real = 1;
    fFail[1].imag = 0;
    /*
    for(int i = 1; i<mRow; i++)
    {
       vcMulByConj(mNumunits, vSelTrans+i*mNumunits, vSelTrans+(i-1)*mNumunits, vXwl);
       xwSum = mSum(vXwl, mNumunits);
       fAngle = xwSum.imag / xwSum.real;
       fFSum += atanf(fAngle);
       fFail[i].real = cosf(-fFSum);
       fFail[i].imag = sinf(-fFSum);
    }
    */
    tbb::parallel_for(tbb::blocked_range<int>(1, mRow), [&](tbb::blocked_range<int> r)
        {
            for (int jj = r.begin(); jj < r.end(); jj++)
            {
                vcMulByConj(mNumunits, vSelTrans + (jj * mNumunits), vSelTrans + ((jj - 1) * mNumunits), vXwl);
                xwSum = mSum(vXwl, mNumunits);
                fAngle = xwSum.imag / xwSum.real;
                fFSum += atanf(fAngle);
                fFail[jj].real = cosf(-fFSum);
                fFail[jj].imag = sinf(-fFSum);
            }
        });
    //参考邱晓辉《ISAR成像快速最小熵相位补偿方法》，电子与信息学报，2004
    //迭代200次
    /*************
    fai_l0 = fai_l;
    tmpData = diag(fai_l) * data;%^ G(n, k) = G(n, k) * exp(-j * ^sita(n));  n:回波数 k : 距离单元个数
    tmpData = fft(tmpData);% I(n, k) = fft(G(n, k));
    tmpData1 = conj(tmpData).*log(abs(tmpData));% ln | (I(q, k)) | *I * (q, k);  q:回波数  k : 距离单元个数
    RL = fft(tmpData1);% 等效于fft(tmpData1, [], 1)
    wn = sum(data.*RL, 2);% w(n) = sigma(G(n, k) * fft(tmpData1, [], 1))
    fai_l = conj(wn). / abs(wn);% exp(-j * ^sita(n)) = w * (n)/|w(n) |
    waitbar(k / search_num, wb);
    compData = diag(fai_l) * data1;
    **************/
    //先构造一个对角矩阵
    MKL_Complex8* vResDta = (MKL_Complex8*)MKL_malloc(mNumunits * mRow * sizeof(MKL_Complex8), 64);
    MKL_Complex8* vResTrans = (MKL_Complex8*)MKL_malloc(mNumunits * mRow * sizeof(MKL_Complex8), 64);
    MKL_Complex8* vResFft = (MKL_Complex8*)MKL_malloc(mNumunits * mRow * sizeof(MKL_Complex8), 64);
    MKL_Complex8* vResFftN = (MKL_Complex8*)MKL_malloc(mNumunits * mRow * sizeof(MKL_Complex8), 64);
    MKL_Complex8* vResRl = (MKL_Complex8*)MKL_malloc(mNumunits * mRow * sizeof(MKL_Complex8), 64);

    float* vResAbs = (float*)MKL_malloc(mNumunits * mRow * sizeof(float), 64);
    float* vResLog = (float*)MKL_malloc(mNumunits * mRow * sizeof(float), 64);

    MKL_Complex8* vWn = (MKL_Complex8*)MKL_malloc(mRow * sizeof(MKL_Complex8), 64);
    MKL_Complex8* vWnConj = (MKL_Complex8*)MKL_malloc(mRow * sizeof(MKL_Complex8), 64);
    float* vWnAbs = (float*)MKL_malloc(mRow * sizeof(float), 64);

    DftiSetValue(fHandle, DFTI_NUMBER_OF_TRANSFORMS, mNumunits);
    DftiCommitDescriptor(fHandle);

    auto tStart_am = std::chrono::high_resolution_clock::now();


    for (int i = 0; i < 100; i++)
    {
        memcpy(fFailCopy, fFail, mRow * sizeof(MKL_Complex8));
        memcpy(vResDta, vSelTrans, mNumunits * mRow * sizeof(MKL_Complex8));
        /*
        for (int j = 0; j < mRow; j++)
            vFDiag[j + j * mRow] = fFail[j];
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mRow, mRow, mNumunits, &alpha, vFDiag, mRow, vSelTrans, mNumunits,
            &alpha, vResDta, mNumunits);
        */
        //PlanB 
        for (int j = 0; j < mRow; j++)
            cblas_cscal(mNumunits, &fFailCopy[j], vResDta + (j * mNumunits), 1);
        //转置
        mkl_comatcopy('R'    /* row-major ordering */,
            'T'    /* matrix will be transposed */,
            mRow     /* rows */,
            mNumunits     /* cols */,
            alpha /* scales the input matrix */,
            vResDta   /* source matrix */,
            mNumunits     /* src_stride */,
            vResTrans   /* destination matrix */,
            mRow     /* dst_stride */);
        DftiComputeForward(fHandle, vResTrans, vResFft);
        vcAbs(mNumunits * mRow, vResFft, vResAbs);
        vsLn(mNumunits * mRow, vResAbs, vResLog);
        vcConj(mNumunits * mRow, vResFft, vResFft);
        tbb::parallel_for(tbb::blocked_range<int>(0, mNumunits * mRow), [&](tbb::blocked_range<int> r)
            {
                for (int jj = r.begin(); jj < r.end(); jj++)
                {
                    vResFftN[jj].real = vResFft[jj].real * vResLog[jj];
                    vResFftN[jj].imag = vResFft[jj].imag * vResLog[jj];
                }
            });
        DftiComputeForward(fHandle, vResFftN, vResRl);
        //转置
        mkl_comatcopy('R'    /* row-major ordering */,
            'T'    /* matrix will be transposed */,
            mNumunits     /* rows */,
            mRow     /* cols */,
            alpha /* scales the input matrix */,
            vResRl   /* source matrix */,
            mRow     /* src_stride */,
            vResFftN   /* destination matrix */,
            mNumunits     /* dst_stride */);
        vcMul(mRow * mNumunits, vSelTrans, vResFftN, vResFftN);
        tbb::parallel_for(tbb::blocked_range<int>(0, mRow), [&](tbb::blocked_range<int> r)
            {
                for (int jj = r.begin(); jj < r.end(); jj++)
                {
                    vWn[jj] = mSum(vResFftN + (jj * mNumunits), mNumunits);
                }
            });
        vcConj(mRow, vWn, vWnConj);
        vcAbs(mRow, vWn, vWnAbs);

        for (int jj = 0; jj < mRow; jj++)
        {
            fFail[jj].real = vWnConj[jj].real / vWnAbs[jj];
            fFail[jj].imag = vWnConj[jj].imag / vWnAbs[jj];
        }
    }
    //相位补偿 
    memcpy(outdta, indta, mCol * mRow * sizeof(MKL_Complex8));
    tbb::parallel_for(tbb::blocked_range<int>(0, mRow), [&](tbb::blocked_range<int> r)
        {
            for (int jj = r.begin(); jj < r.end(); jj++)
            {
                cblas_cscal(mCol, &fFail[jj], outdta + (jj * mCol), 1);
            }
        });

    auto tEnd_afc = std::chrono::high_resolution_clock::now();
    std::cout << "************************************" << std::endl;
    std::cout << "Time consumption of fast-entropy autofocusing: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_afc - tStart_am).count()
        << "ms" << std::endl;
    std::cout << "Fast-entropy autofocusing has been done!" << std::endl;
    std::cout << "************************************\n" << std::endl;

    DftiFreeDescriptor(&fHandle);
    MKL_free(vTmpdta);
    MKL_free(fFail);
    MKL_free(fFailCopy);
    MKL_free(vTmptrans);
    MKL_free(vMax);
    MKL_free(vTmptransAbs);
    MKL_free(vMaxIdx);
    MKL_free(vImgAbs);
    MKL_free(vImgSqr);
    MKL_free(vSum);
    MKL_free(vSumCopy);
    MKL_free(vSeIdx);
    MKL_free(vNormal);
    MKL_free(vNormalLg);
    MKL_free(vEntroBase);
    MKL_free(vEntropy);
    MKL_free(vSelDta);
    MKL_free(vSeIdxB);
    MKL_free(vXwl);
    MKL_free(vResDta);
    MKL_free(vResTrans);
    MKL_free(vResFft);
    MKL_free(vResAbs);
    MKL_free(vResLog);
    MKL_free(vResFftN);
    MKL_free(vResRl);
    MKL_free(vWn);
    MKL_free(vWnConj);
    MKL_free(vWnAbs);
}

void FastKTA(MKL_Complex8* indta, MKL_Complex8* outDta, RadarParameters& mPara)
{
    //转置
    MKL_Complex8 alpha;
    alpha.real = 1.;
    alpha.imag = 0.;
    size_t mRow = mPara.num_echoes;
    size_t mCol = mPara.cLen;

    const double fpose = mPara.kai * mPara.Tp / (mPara.num_range_bins - 1) / mPara.fc;
    const double ftwo = 1 - double(mPara.kai * mPara.Tp / 2 / mPara.fc);
    const int mRange = mPara.num_range_bins;
    const int mDopp = mPara.num_echoes;
    const int vlen = nextPow2(mPara.num_echoes * 2 - 1);
    //循环对每一个距离单元做CZT变换
    MKL_Complex8* pMov = nullptr;
    MKL_Complex8* pRangeVec = (MKL_Complex8*)MKL_malloc(sizeof(MKL_Complex8) * mDopp, 64);
    MKL_Complex8* pRangeWn = (MKL_Complex8*)MKL_malloc(sizeof(MKL_Complex8) * vlen, 64);
    MKL_Complex8* pRangeWnA = (MKL_Complex8*)MKL_malloc(sizeof(MKL_Complex8) * vlen * mRange, 64);
    MKL_Complex8* pRangeAn = (MKL_Complex8*)MKL_malloc(sizeof(MKL_Complex8) * mDopp, 64);
    MKL_Complex8* pRangeAan = (MKL_Complex8*)MKL_malloc(sizeof(MKL_Complex8) * mDopp * mRange, 64);
    MKL_Complex8* pY = (MKL_Complex8*)MKL_malloc(sizeof(MKL_Complex8) * mDopp * mRange, 64);
    MKL_Complex8* pYfft = (MKL_Complex8*)MKL_calloc(sizeof(MKL_Complex8), vlen * mRange, 64);
    MKL_Complex8* pWnOne = (MKL_Complex8*)MKL_calloc(sizeof(MKL_Complex8), vlen, 64);
    MKL_Complex8* pWnFft = (MKL_Complex8*)MKL_calloc(sizeof(MKL_Complex8), vlen * mRange, 64);
    MKL_Complex8* pWnInv = (MKL_Complex8*)MKL_calloc(sizeof(MKL_Complex8), vlen, 64);
    MKL_Complex8* pYY = (MKL_Complex8*)MKL_calloc(sizeof(MKL_Complex8), vlen, 64);
    MKL_Complex8* pRes = (MKL_Complex8*)MKL_malloc(sizeof(MKL_Complex8) * mRange * mDopp, 64);
    MKL_Complex8* pResSft = (MKL_Complex8*)MKL_malloc(sizeof(MKL_Complex8) * mRange * mDopp, 64);
    double fker = 0;
    double fker2 = 0;
    MKL_Complex16 comW = { 0.0f,0.0f };
    MKL_Complex8 comFai = { 0.0f,0.0f };
    //FFT
    //FFT Forward
    DFTI_DESCRIPTOR_HANDLE fHandle = NULL;
    DftiCreateDescriptor(&fHandle, DFTI_SINGLE, DFTI_COMPLEX, 1, vlen);
    DftiSetValue(fHandle, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(fHandle, DFTI_NUMBER_OF_TRANSFORMS, mRange);
    //每次FFT距离
    DftiSetValue(fHandle, DFTI_INPUT_DISTANCE, vlen);
    DftiSetValue(fHandle, DFTI_OUTPUT_DISTANCE, vlen);
    DftiSetValue(fHandle, DFTI_FORWARD_SCALE, 1.0);
    DftiSetValue(fHandle, DFTI_BACKWARD_SCALE, 1.0 / vlen);
    DftiCommitDescriptor(fHandle);
    //handle 2
    DFTI_DESCRIPTOR_HANDLE fHandleA = NULL;
    DftiCreateDescriptor(&fHandleA, DFTI_SINGLE, DFTI_COMPLEX, 1, vlen);
    DftiSetValue(fHandleA, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(fHandleA, DFTI_FORWARD_SCALE, 1.0);
    DftiSetValue(fHandleA, DFTI_BACKWARD_SCALE, 1.0 / vlen);
    DftiCommitDescriptor(fHandleA);
    //dopple
    DFTI_DESCRIPTOR_HANDLE fHandleB = NULL;
    DftiCreateDescriptor(&fHandleB, DFTI_SINGLE, DFTI_COMPLEX, 1, mDopp);
    DftiSetValue(fHandleB, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(fHandleB, DFTI_NUMBER_OF_TRANSFORMS, mRange);
    //每次FFT距离
    DftiSetValue(fHandleB, DFTI_INPUT_DISTANCE, mDopp);
    DftiSetValue(fHandleB, DFTI_OUTPUT_DISTANCE, mDopp);
    DftiSetValue(fHandleB, DFTI_FORWARD_SCALE, 1.0);
    DftiSetValue(fHandleB, DFTI_BACKWARD_SCALE, 1.0 / mDopp);
    DftiCommitDescriptor(fHandleB);


    std::vector<float> mSeq(mDopp * 2);
    //构造递增数列
    std::iota(mSeq.begin(), mSeq.end(), -mDopp + 1);
    //求Square
    float* pRangeSqr = (float*)MKL_malloc(sizeof(float) * vlen, 64);
    float* pSeq = reinterpret_cast<float*>(mSeq.data());
    vsSqr(mDopp, pSeq, pRangeSqr);
    //构造第二个递增数列
    std::vector<float> mSeqb(mDopp);
    std::iota(mSeqb.begin(), mSeqb.end(), 0);
    //构造一个全1的
    std::for_each(pWnOne, pWnOne + mDopp * 2 - 1, [&](MKL_Complex8& mCplx) {
        mCplx.real = 1.0f;
    mCplx.imag = 0.0f;
        });
    for (int i = 0; i < mRange; i++)
    {
        fker = -2 * M_PI * (ftwo + fpose * i) / mDopp;
        comW.real = cos(fker);
        comW.imag = sin(fker);
        fker2 = fker * mDopp / 2;
        comFai.real = float(cos(fker));
        comFai.imag = float(sin(fker));
        //因为pRangeSqr是对称的，所以只计算一半
        //改成用double计算
        //比较求幂次从N开始 
        //计算因子
        //改为double
        //先求相位
        std::transform(pRangeSqr, pRangeSqr + mDopp, pRangeWn, [&](float wa)-> MKL_Complex8 {
            double num = wa / 2;
        double nPhase = num * fker;
        double cRe = cos(nPhase);
        double cIma = sin(nPhase);
        MKL_Complex8 ww = { float(cRe), float(cIma) };
        return ww;
            });
        for (int idx = 1; idx < mDopp; idx++)
        {
            pRangeWn[mDopp + idx - 1] = pRangeWn[mDopp - idx - 1];
        }
        memcpy(pRangeWnA + i * vlen, pRangeWn, vlen * sizeof(double));
        //aa = a .^ ( -nn );
        //aa = aa.*ww(m + nn);
        std::transform(mSeqb.begin(), mSeqb.end(), pRangeAn, [&](float aa)->MKL_Complex8 {
            double nPhase = -aa * fker2;
        double cRe = cos(nPhase);
        double cIma = sin(nPhase);
        MKL_Complex8 ww = { float(cRe), float(cIma) };
        return ww;
            });
        vcMul(mDopp, pRangeAn, pRangeWn + mDopp - 1, pRangeAan + i * mDopp);
        vcDiv(mDopp * 2 - 1, pWnOne, pRangeWn, pWnInv);
        DftiComputeForward(fHandleA, pWnInv);
        memcpy(pWnFft + i * vlen, pWnInv, vlen * sizeof(double));
        memset(pWnInv, 0, sizeof(double) * vlen);
    }

    auto tStart_RA = std::chrono::high_resolution_clock::now();


    vcMul(mDopp * mRange, pRangeAan, indta, pY);
    tbb::parallel_for(tbb::blocked_range<int>(0, mRange), [&](tbb::blocked_range<int> r)
        {
            for (int jj = r.begin(); jj < r.end(); jj++)
            {
                memcpy(pYfft + jj * vlen, pY + jj * mDopp, mDopp * sizeof(double));
            }
        });
    DftiComputeForward(fHandle, pYfft);
    vcMul(vlen * mRange, pWnFft, pYfft, pYfft);
    DftiComputeBackward(fHandle, pYfft);
    tbb::parallel_for(tbb::blocked_range<int>(0, mRange), [&](tbb::blocked_range<int> r)
        {
            for (int jj = r.begin(); jj < r.end(); jj++)
            {
                vcMul(mDopp, pYfft + jj * vlen + mDopp - 1, pRangeWnA + jj * vlen + mDopp - 1, pRes + mDopp * jj);
                //fftshift(xn)
                memcpy(pResSft + mDopp * jj, pRes + mDopp * jj + mDopp / 2, sizeof(double) * mDopp / 2);
                memcpy(pResSft + mDopp * jj + mDopp / 2,
                    pRes + mDopp * jj, sizeof(double) * mDopp / 2);
            }
        });
    DftiComputeBackward(fHandleB, pResSft);
    //shift
    tbb::parallel_for(tbb::blocked_range<int>(0, mRange), [&](tbb::blocked_range<int> r)
        {
            for (int jj = r.begin(); jj < r.end(); jj++)
            {
                //fftshift(xn)
                memcpy(pRes + mDopp * jj, pResSft + mDopp * jj + mDopp / 2, sizeof(double) * mDopp / 2);
                memcpy(pRes + mDopp * jj + mDopp / 2,
                    pResSft + mDopp * jj, sizeof(double) * mDopp / 2);
            }
        });
    //S_key=S_key.'
    mkl_comatcopy('R'    /* row-major ordering */,
        'T'    /* matrix will be transposed */,
        mRange     /* rows */,
        mDopp     /* cols */,
        alpha /* scales the input matrix */,
        pRes   /* source matrix */,
        mRow     /* src_stride */,
        pResSft   /* destination matrix */,
        mRange     /* dst_stride */);
    //S_key=fftshift(fft(S_key,[],2),2);
    DftiCreateDescriptor(&fHandleB, DFTI_SINGLE, DFTI_COMPLEX, 1, mRange);
    DftiSetValue(fHandleB, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(fHandleB, DFTI_NUMBER_OF_TRANSFORMS, mDopp);
    //每次FFT距离
    DftiSetValue(fHandleB, DFTI_INPUT_DISTANCE, mRange);
    DftiSetValue(fHandleB, DFTI_OUTPUT_DISTANCE, mRange);
    DftiSetValue(fHandleB, DFTI_FORWARD_SCALE, 1.0);
    DftiCommitDescriptor(fHandleB);

    DftiComputeForward(fHandleB, pResSft);

    //shift
    tbb::parallel_for(tbb::blocked_range<int>(0, mDopp), [&](tbb::blocked_range<int> r)
        {
            for (int jj = r.begin(); jj < r.end(); jj++)
            {
                //fftshift(xn)
                memcpy(pRes + mRange * jj, pResSft + mRange * jj + mRange / 2, sizeof(double) * mRange / 2);
                memcpy(pRes + mRange * jj + mRange / 2,
                    pResSft + mRange * jj, sizeof(double) * mRange / 2);
                //S_key = fliplr(S_key);
                /*for (int i = 0; i < mRange/2; i++)
                {
                    MKL_Complex8 temp = pRes[i + mRange * jj];
                    pRes[i + mRange * jj] = pRes[i + mRange * jj + mRange - i];
                    pRes[i + mRange * jj + mRange - i] = temp;
                }*/
            }
        });
    //再沿着方向位Shift
    tbb::parallel_for(tbb::blocked_range<int>(0, mDopp), [&](tbb::blocked_range<int> r)
        {
            for (int jj = r.begin(); jj < r.end(); jj++)
            {
                if (jj < mDopp / 2)
                {
                    //fftshift(xn)
                    memcpy(pResSft + mRange * (jj + mDopp / 2)
                        , pRes + mRange * jj, sizeof(double) * mRange);
                }
                else
                {
                    memcpy(pResSft + mRange * (jj - mDopp / 2)
                        , pRes + mRange * jj, sizeof(double) * mRange);
                }
            }

        });
    //S_key = fliplr(S_key);% 将变换结果上下翻转，与（*** ）行代码配套使用

    auto tEnd_RA = std::chrono::high_resolution_clock::now();
    std::cout << "************************************" << std::endl;
    std::cout << "Time consumption of dechirp: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(tEnd_RA - tStart_RA).count()
        << "ms" << std::endl;
    std::cout << "keystone has been done!" << std::endl;
    std::cout << "************************************\n" << std::endl;

    mkl_free(pRangeSqr);
    mkl_free(pRangeVec);
    mkl_free(pRangeWn);
    mkl_free(pRangeWnA);
    mkl_free(pRangeAn);
    mkl_free(pRangeAan);
    mkl_free(pY);
    mkl_free(pYfft);
    mkl_free(pWnOne);
    mkl_free(pWnInv);
    mkl_free(pYY);
    mkl_free(pRes);
    mkl_free(pWnFft);
    mkl_free(pResSft);
    DftiFreeDescriptor(&fHandle);
    DftiFreeDescriptor(&fHandleA);
    DftiFreeDescriptor(&fHandleB);
}

void DopplerProcessA(MKL_Complex8* indta, float* OutImg, RadarParameters& mPara)
{
    //转置
    MKL_Complex8 alpha;
    alpha.real = 1.;
    alpha.imag = 0.;
    size_t mRow = mPara.num_echoes;
    size_t mCol = mPara.cLen;
    //先横向加窗
    float* mHamming = (float*)MKL_malloc(mRow * sizeof(float), 32);
    Hamming(mRow, mHamming);

    MKL_Complex8* mHamCplx = (MKL_Complex8*)MKL_malloc(mRow * sizeof(MKL_Complex8), 64);
    //加窗之后的数据
    MKL_Complex8* WinHam = (MKL_Complex8*)MKL_malloc(mRow * mCol * sizeof(MKL_Complex8), 64);
    //转置后的数据
    MKL_Complex8* TransDta = (MKL_Complex8*)MKL_malloc(mRow * mCol * sizeof(MKL_Complex8), 64);
    //FFT之后的数据
    MKL_Complex8* TransDtaFft = (MKL_Complex8*)MKL_malloc(mRow * mCol * sizeof(MKL_Complex8), 64);

    for (int i = 0; i < mRow; i++)
    {
        mHamCplx[i].real = mHamming[i] * cosf(i * PI_h);
        mHamCplx[i].imag = 0;
    }

    //先转置
    //转置
    mkl_comatcopy('R'    /* row-major ordering */,
        'T'    /* matrix will be transposed */,
        mRow     /* rows */,
        mCol     /* cols */,
        alpha /* scales the input matrix */,
        indta   /* source matrix */,
        mCol     /* src_stride */,
        TransDta   /* destination matrix */,
        mRow     /* dst_stride */);

    tbb::parallel_for(tbb::blocked_range<int>(0, mCol), [&](tbb::blocked_range<int> r)
        {
            for (int jj = r.begin(); jj < r.end(); jj++)
            {
                vcMul(mRow, mHamCplx, TransDta + jj * mRow, WinHam + jj * mRow);
            }
        });

    //准备FFT
    //FFT Forward
    DFTI_DESCRIPTOR_HANDLE fHandle = NULL;
    DftiCreateDescriptor(&fHandle, DFTI_SINGLE, DFTI_COMPLEX, 1, mRow);
    DftiSetValue(fHandle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    //多次FFT
    DftiSetValue(fHandle, DFTI_NUMBER_OF_TRANSFORMS, mCol);
    //每次FFT距离
    DftiSetValue(fHandle, DFTI_INPUT_DISTANCE, mRow);
    DftiSetValue(fHandle, DFTI_OUTPUT_DISTANCE, mRow);
    //FFT Scale
    DftiSetValue(fHandle, DFTI_FORWARD_SCALE, 1.0);

    DftiCommitDescriptor(fHandle);
    DftiComputeForward(fHandle, WinHam, TransDtaFft);
    //再转置为方向维和距离维
    mkl_comatcopy('R'    /* row-major ordering */,
        'T'    /* matrix will be transposed */,
        mCol     /* rows */,
        mRow     /* cols */,
        alpha /* scales the input matrix */,
        TransDtaFft   /* source matrix */,
        mRow     /* src_stride */,
        TransDta   /* destination matrix */,
        mCol     /* dst_stride */);

    vcAbs(mCol * mRow, TransDta, OutImg);

    DftiFreeDescriptor(&fHandle);
    MKL_free(mHamming);
    MKL_free(mHamCplx);
    MKL_free(WinHam);
    MKL_free(TransDta);
    MKL_free(TransDtaFft);
}


