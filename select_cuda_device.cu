#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "select_cuda_device.h"

#define LOCAL					static


LOCAL void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
	const int line) {
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err) {
		fprintf(stderr,
			"%s(%i) : getLastCudaError() CUDA error :"
			" %s : (%d) %s.\n",
			file, line, errorMessage, static_cast<int>(err),
			cudaGetErrorString(err));
		cudaDeviceReset();
			exit(EXIT_FAILURE);
	}
}


LOCAL __global__ void WarmUpGPU(int dummy_data_len, int *p_dummy_data)
{
	int i;

	i = threadIdx.x;	
	while (i < dummy_data_len)
	{
		int temp;
		temp = p_dummy_data[i] * p_dummy_data[i];
		p_dummy_data[i] = temp;
		i += blockDim.x;
	}/*while*/

}/*WarmUpGPU*/

 //copy from helper_cuda.h
 // Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor) {
	// Defines for GPU Architecture types (using the SM version to determine
	// the # of cores per SM
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
				 // and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{ 0x30, 192 },
		{ 0x32, 192 },
		{ 0x35, 192 },
		{ 0x37, 192 },
		{ 0x50, 128 },
		{ 0x52, 128 },
		{ 0x53, 128 },
		{ 0x60,  64 },
		{ 0x61, 128 },
		{ 0x62, 128 },
		{ 0x70,  64 },
		{ 0x72,  64 },
		{ 0x75,  64 },
		{ -1, -1 } };

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one
	// to run properly
	printf(
		"MapSMtoCores for SM %d.%d is undefined."
		"  Default to use %d Cores/SM\n",
		major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	return nGpuArchCoresPerSM[index - 1].Cores;
}



LOCAL void RunCUDAOnce(void)
{
	int dummy_data_len;
	int *p_dummy_data_dev;

	printf("the CUDA device has warmed up\r\n");
	dummy_data_len = 1024;
	HANDLE_ERROR(cudaMalloc((void**)&p_dummy_data_dev,
		dummy_data_len * sizeof(int)));

	cudaMemset(p_dummy_data_dev, 1, dummy_data_len * sizeof(int));

	for (int i = 0; i < 3; i++)
		WarmUpGPU << < 32, 32 >> > (dummy_data_len, p_dummy_data_dev);
	getLastCudaError("WarmUpGPU");
	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaFree(p_dummy_data_dev)); p_dummy_data_dev = NULL;

}/*RunCUDAOnce*/


int SelectCudaDevice(void)
{
	int device_number;
	int candidate_id;
	double candidate_performance;

	// If the command-line has a device number specified, use it
	// Otherwise pick the device with highest Gflops/s
	cudaGetDeviceCount(&device_number);
	if (0 == device_number)
		return -1;

	candidate_performance = 0.0;

	for (int i = 0; i < device_number; i++) {
		cudaDeviceProp prop;
		int number_of_cores;
		cudaGetDeviceProperties(&prop, i);

		number_of_cores = _ConvertSMVer2Cores(prop.major, prop.minor)
			*prop.multiProcessorCount;
		double performance =
			prop.clockRate*number_of_cores*
			prop.memoryClockRate * 1000.0
			*prop.memoryBusWidth / 8 / 1024 / 1024 / 1024.0;
		if (performance > candidate_performance)
		{
			candidate_id = i;
			candidate_performance = performance;
		}/*if */
	}

	cudaSetDevice(candidate_id);
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, candidate_id);
		printf("GPU Device : \"%s\" has been selected:: \r\n",
			prop.name);
		printf("\tcompute capability = %d.%d\r\n", prop.major, prop.minor);
		printf("\t%d cores in %.3f GHz\r\n",
			_ConvertSMVer2Cores(prop.major, prop.minor)
			*prop.multiProcessorCount, prop.clockRate / 1000.0 / 1009.0);

#define TWO_WAYS				(2)
		printf("\tbus = %d bit, bandwidth = %3.1f GB/s\r\n", prop.memoryBusWidth,
			TWO_WAYS*prop.memoryClockRate * 1000.0
			*prop.memoryBusWidth / 8 / 1024 / 1024 / 1024.0);
		printf("\tL2 cache size = %d KB\r\n", prop.l2CacheSize / 1024);
	}
	RunCUDAOnce();

	return 0;
}/*findCudaDevice*/
