#include "HighSpeedCompensation.cuh"


/*****************************************
 * 函数功能：对输入回波进行高速运动补偿
 * 输入参数：
 * d_data：  回波（按回波依次存入内存）
 * Fs:       采样频率
 * fc：      载频
 * band_width:
			 带宽
 * Tp:
			 脉宽
 * velocity_data:
			 慢时间对应的速度
 * range_data:
			 慢时间对应的距离
 * echo_num：回波数
 * range_num:距离单元数
 * 原理说明：
 * 依据雷达系统参数和速度、距离信息构建补偿相位：
 * fai = 2*pi*K*2*v_tmp/c*tk.^2 + 4*pi*f0*v_tmp/c*tk - 4*pi*K/c^2*(2*Range(i)*v_tmp*tk + v_tmp^2*tk.^2);  (*)
 * 根据相位构建补偿函数：
 * exp(1j*fai)
 *******************************************/
void HighSpeedCompensation(cuComplex* d_data, unsigned int Fs, long long fc, long long band_width, float Tp, float* velocity_data, float* range_data, int echo_num, int range_num, cublasHandle_t handle_HSC)
{

	thrust::device_vector<float>t_fast(range_num);
	float t_fast_start = -float(range_num) / 2.0;                                                               // 构造快时间向量tk=[-N/2:N/2-1]/fs
	thrust::sequence(thrust::device, t_fast.begin(), t_fast.end(), t_fast_start);
	thrust::transform(thrust::device, t_fast.begin(), t_fast.end(), t_fast.begin(), nor_by_fs(Fs));
	float* d_t_fast = reinterpret_cast<float*>(thrust::raw_pointer_cast(t_fast.data()));                        // 类型转换：thrust->cu
	thrust::device_vector<float>t_fast_2(range_num);                                                            // 计算tk^2
	thrust::transform(thrust::device, t_fast.begin(), t_fast.end(), t_fast_2.begin(), square_functor<float>());
	float* d_t_fast_2 = reinterpret_cast<float*>(thrust::raw_pointer_cast(t_fast_2.data()));                    // 类型转换：thrust->cu

	// ====================================== //
	// * 有待修改
	// 这里是将速度、距离分开传输
	// 实际应用时可以将所有窄带信息连同回波放在一个连续的地址
	float* d_velocity;
	checkCudaErrors(cudaMalloc((void**)&d_velocity, sizeof(float) * echo_num));
	checkCudaErrors(cudaMemcpy(d_velocity, velocity_data, sizeof(float) * echo_num, cudaMemcpyHostToDevice));       // 速度数据传到设备上
	thrust::device_ptr<float>thr_d_velocity(d_velocity);                                                        // 类型转换：cu->thrust
	thrust::device_vector<float>velocity_2(echo_num);                                                           // 计算v^2
	thrust::transform(thrust::device, thr_d_velocity, thr_d_velocity + echo_num, velocity_2.begin(), square_functor<float>());
	float* d_velocity_2 = reinterpret_cast<float*>(thrust::raw_pointer_cast(velocity_2.data()));                // 类型转换：thrust->cu

	float* d_range;
	checkCudaErrors(cudaMalloc((void**)&d_range, sizeof(float) * echo_num));
	checkCudaErrors(cudaMemcpy(d_range, range_data, sizeof(float) * echo_num, cudaMemcpyHostToDevice));
	thrust::device_ptr<float>thr_d_range(d_range);                                                              // 类型转换：cu->thrust 

	thrust::device_vector<float>thr_r_dot_v(echo_num);                                                          // 计算r 点乘 v
	thrust::transform(thrust::device, thr_d_range, thr_d_range + echo_num, thr_d_velocity, thr_r_dot_v.begin(), thrust::multiplies<float>());
	float* r_dot_v = reinterpret_cast<float*>(thrust::raw_pointer_cast(thr_r_dot_v.data()));                    // 类型转换：thrust->cu
	// ====================================== //

	float chirp_rate = float(band_width) / Tp;                                                                  // 三个系数，参考(*)式
	float coefficient1 = 4.0 * PI_h * chirp_rate / lightSpeed_h;
	float coefficient2 = float(4 * PI_h * fc) / lightSpeed_h;
	float coefficient3 = -4.0 * PI_h * chirp_rate / float(float(lightSpeed_h) * float(lightSpeed_h));
	float coefficient4 = -8.0 * PI_h * chirp_rate / float(float(lightSpeed_h) * float(lightSpeed_h));

	float* v_mul_tk2;                                                                                           // 计算(*)中的四项
	checkCudaErrors(cudaMalloc((void**)&v_mul_tk2, sizeof(float) * echo_num * range_num));
	checkCudaErrors(cudaMemset(v_mul_tk2, 0.0, sizeof(float) * echo_num * range_num));

	float* v_mul_tk;
	checkCudaErrors(cudaMalloc((void**)&v_mul_tk, sizeof(float) * echo_num * range_num));
	checkCudaErrors(cudaMemset(v_mul_tk, 0.0, sizeof(float) * echo_num * range_num));

	float* v2_mul_tk2;
	checkCudaErrors(cudaMalloc((void**)&v2_mul_tk2, sizeof(float) * echo_num * range_num));
	checkCudaErrors(cudaMemset(v2_mul_tk2, 0.0, sizeof(float) * echo_num * range_num));

	float* r_mul_v_mul_tk;
	checkCudaErrors(cudaMalloc((void**)&r_mul_v_mul_tk, sizeof(float) * echo_num * range_num));
	checkCudaErrors(cudaMemset(r_mul_v_mul_tk, 0.0, sizeof(float) * echo_num * range_num));

	// *修改20200628：交换5th和7th两项;1th和2th两项；最后一项由echo_num->range_num
	checkCublasErrors(cublasSger(handle_HSC, range_num, echo_num, &coefficient1, d_t_fast_2, 1, d_velocity, 1, v_mul_tk2, range_num));
	checkCublasErrors(cublasSger(handle_HSC, range_num, echo_num, &coefficient2, d_t_fast, 1, d_velocity, 1, v_mul_tk, range_num));
	checkCublasErrors(cublasSger(handle_HSC, range_num, echo_num, &coefficient3, d_t_fast_2, 1, d_velocity_2, 1, v2_mul_tk2, range_num));
	checkCublasErrors(cublasSger(handle_HSC, range_num, echo_num, &coefficient4, d_t_fast, 1, r_dot_v, 1, r_mul_v_mul_tk, range_num));

	thrust::device_ptr<float>thr_v_mul_tk2(v_mul_tk2);                                                           // 四项累加，计算补偿相位
	thrust::device_ptr<float>thr_v_mul_tk(v_mul_tk);
	thrust::device_ptr<float>thr_v2_mul_tk2(v2_mul_tk2);
	thrust::device_ptr<float>thr_r_mul_v_mul_tk(r_mul_v_mul_tk);
	thrust::device_vector<float>thr_phase(echo_num * range_num);

	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(thr_v_mul_tk2, thr_v_mul_tk, thr_v2_mul_tk2, thr_r_mul_v_mul_tk, thr_phase.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(thr_v_mul_tk2 + echo_num * range_num,
			thr_v_mul_tk + echo_num * range_num, thr_v2_mul_tk2 + echo_num * range_num, thr_r_mul_v_mul_tk + echo_num * range_num, thr_phase.end())),
		arbitrary_functor_four_sum());

	thrust::device_vector<comThr>thr_phi(echo_num * range_num);                                                   // 计算补偿函数
	thrust::transform(thrust::device, thr_phase.begin(), thr_phase.end(), thr_phi.begin(), getPhi_functor());

	comThr* thr_temp_d_Data = reinterpret_cast<comThr*>(d_data);                                                // 高速运动补偿
	thrust::device_ptr<comThr> thr_d_data = thrust::device_pointer_cast(thr_temp_d_Data);
	Complex_Mul_Complex op_comMul;
	thrust::transform(thrust::device, thr_d_data, thr_d_data + echo_num * range_num, thr_phi.begin(), thr_d_data, op_comMul);

	// 释放空间
	checkCudaErrors(cudaFree(d_range));
	checkCudaErrors(cudaFree(d_velocity));
	checkCudaErrors(cudaFree(v_mul_tk2));
	checkCudaErrors(cudaFree(v_mul_tk));
	checkCudaErrors(cudaFree(v2_mul_tk2));
	checkCudaErrors(cudaFree(r_mul_v_mul_tk));

	//checkCublasErrors(cublasDestroy(handle_HSC));
}