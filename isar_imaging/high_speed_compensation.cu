#include "high_speed_compensation.cuh"

void highSpeedCompensation(cuComplex* d_data, int Fs, long long band_width, float Tp, float* h_velocity, int echo_num, int range_num, cublasHandle_t handle)
{
	int data_num = echo_num * range_num;

	thrust::device_vector<float> tk(range_num);  // fast time vector: tk=[0:N-1]/fs
	thrust::sequence(thrust::device, tk.begin(), tk.end(), 0.0f);
	thrust::transform(thrust::device, tk.begin(), tk.end(), tk.begin(), [=]__host__ __device__(const float& x) { return x / static_cast<float>(Fs); });
	float* d_tk = reinterpret_cast<float*>(thrust::raw_pointer_cast(tk.data()));  // type cast: thrust -> float
	
	thrust::device_vector<float> tk2(range_num);  // calculate tk^2
	thrust::transform(thrust::device, tk.begin(), tk.end(), tk2.begin(), []__host__ __device__(const float& x) { return x * x; });
	float* d_tk2 = reinterpret_cast<float*>(thrust::raw_pointer_cast(tk2.data()));

	float* d_velocity = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_velocity, sizeof(float) * echo_num));
	checkCudaErrors(cudaMemcpy(d_velocity, h_velocity, sizeof(float) * echo_num, cudaMemcpyHostToDevice));
	thrust::device_ptr<float> thr_d_velocity(d_velocity);

	float chirp_rate = -static_cast<float>(band_width) / Tp;  // extra minus symbol for velocity (depending on different radar signal)
	float coefficient = 4.0f * PI_h * chirp_rate / lightSpeed_h;  // 4 * pi * K / c

	float* v_mul_tk2 = nullptr;
	checkCudaErrors(cudaMalloc((void**)&v_mul_tk2, sizeof(float) * data_num));
	checkCudaErrors(cudaMemset(v_mul_tk2, 0, sizeof(float) * data_num));

	// coef * v * tk.^2
	checkCudaErrors(cublasSger(handle, range_num, echo_num, &coefficient, d_tk2, 1, d_velocity, 1, v_mul_tk2, range_num));

	thrust::device_ptr<float> thr_v_mul_tk2(v_mul_tk2);
	thrust::device_vector<float> thr_phase(thr_v_mul_tk2, thr_v_mul_tk2 + data_num);

	thrust::device_vector<comThr> thr_phi(data_num);
	thrust::transform(thrust::device, thr_phase.begin(), thr_phase.end(), thr_phi.begin(), []__host__ __device__(const float& x) { return thrust::exp(comThr(0.0f, x)); });

	thrust::device_ptr<comThr> thr_d_data = thrust::device_pointer_cast(reinterpret_cast<comThr*>(d_data));
	thrust::transform(thrust::device, thr_d_data, thr_d_data + data_num, thr_phi.begin(), thr_d_data, \
		[]__host__ __device__(const comThr& a, const comThr& b) { return a * b; });

	// free gpu mallocated space
	checkCudaErrors(cudaFree(d_velocity));
	checkCudaErrors(cudaFree(v_mul_tk2));

}