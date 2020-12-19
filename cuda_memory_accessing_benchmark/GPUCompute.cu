#include <stdlib.h>
#include "cuda_runtime.h"
#include "GPUCompute.cuh"



#define THREADS_IN_BLOCK					(1024)
#define NUM_BLOCKS							(16)

#define UNUSED(EXPR)						do { (void)(EXPR); } while (0)

//#define _BEST_BLOCK_NUMBER
/**********************************************************************/

__global__ void KernelAdd(float *p_input1, float *p_input2, int length, 
	float *p_output)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (; i < length; i += gridDim.x * blockDim.x)
		p_output[i] = p_input1[i] + p_input2[i];
}

/**********************************************************************/

__global__ void KernelMultiple(float *p_input1, float *p_input2, int length, 
	float *p_output)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (; i < length; i += gridDim.x * blockDim.x)
		p_output[i] = p_input1[i] * p_input2[i];
}

/**********************************************************************/

__global__ void KernelMultipleByConstant(float *p_input, float value,
	int length, float *p_output)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (; i < length; i += gridDim.x * blockDim.x)
		p_output[i] = p_input[i] * value;
}

/**********************************************************************/

/**********************************************************************/
typedef struct 
{
	cudaStream_t stream;

	float *p_dev_input1;
	float *p_dev_input2;
	float *p_dev_working;

	cudaEvent_t start_including_copy;
	cudaEvent_t stop_including_copy;

	cudaEvent_t start_excluding_copy;
	cudaEvent_t stop_excluding_copy;

} GPUComputeHandle;

/**********************************************************************/

int GPUSAXPYSynchronousDeepCopy(CUDAHandle handle, int length, float A,
	float *p_input1, float *p_input2)
{
	GPUComputeHandle *p_compute_handle;
	p_compute_handle = (GPUComputeHandle*)handle;

	if (NULL == p_compute_handle)
		return -1;

	float *p_host_input1 = p_input1;
	float *p_host_input2 = p_input2;

	cudaStream_t stream  = p_compute_handle->stream;
	float *p_dev_input1  = p_compute_handle->p_dev_input1;
	float *p_dev_input2  = p_compute_handle->p_dev_input2;
	float *p_dev_working = p_compute_handle->p_dev_working;
	
	cudaEventRecord(p_compute_handle->start_including_copy, stream);

	cudaMemcpy(p_dev_input1, p_host_input1, length * sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(p_dev_input2, p_host_input2, length * sizeof(float),
		cudaMemcpyHostToDevice);	

	cudaEventRecord(p_compute_handle->start_excluding_copy, stream);

#ifdef _BEST_BLOCK_NUMBER
	int block_num;
	block_num = (length + (THREADS_IN_BLOCK - 1)) / THREADS_IN_BLOCK;
	block_num *= THREADS_IN_BLOCK;
#else
	int block_num = NUM_BLOCKS;
#endif

	KernelMultipleByConstant << < THREADS_IN_BLOCK, block_num, 0, stream >> > (
		p_dev_input1, A, length, p_dev_working);

	KernelAdd << < THREADS_IN_BLOCK, block_num, 0, stream >> > (
		p_dev_input2, p_dev_working, length, p_dev_input2);

	cudaEventRecord(p_compute_handle->stop_excluding_copy, stream);

	cudaMemcpy(p_host_input2, p_dev_input2, length * sizeof(float),
		cudaMemcpyDeviceToHost);

	cudaEventRecord(p_compute_handle->stop_including_copy, stream);

	cudaEventSynchronize(p_compute_handle->stop_including_copy);

	return 0;
}

/**********************************************************************/

int GPUSAXPYSynchronousZeroCopy(CUDAHandle handle, int length, float A,
	float *p_input1, float *p_input2)
{
	GPUComputeHandle *p_compute_handle;
	p_compute_handle = (GPUComputeHandle*)handle;

	if (NULL == p_compute_handle)
		return -1;

	float *p_host_input1 = p_input1;
	float *p_host_input2 = p_input2;

	cudaStream_t stream = p_compute_handle->stream;
	float *p_dev_input1 = p_compute_handle->p_dev_input1;
	float *p_dev_input2 = p_compute_handle->p_dev_input2;
	float *p_dev_working = p_compute_handle->p_dev_working;

	cudaEventRecord(p_compute_handle->start_including_copy, stream);

	cudaHostGetDevicePointer(&p_dev_input1, p_host_input1, 0);
	cudaHostGetDevicePointer(&p_dev_input2, p_host_input2, 0);

	cudaEventRecord(p_compute_handle->start_excluding_copy, stream);

#ifdef _BEST_BLOCK_NUMBER
	int block_num;
	block_num = (length + (THREADS_IN_BLOCK - 1)) / THREADS_IN_BLOCK;
	block_num *= THREADS_IN_BLOCK;
#else
	int block_num = NUM_BLOCKS;
#endif

	KernelMultipleByConstant << < THREADS_IN_BLOCK, block_num, 0, stream >> > (
		p_dev_input1, A, length, p_dev_working);

	KernelAdd << < THREADS_IN_BLOCK, block_num, 0, stream >> > (
		p_dev_input2, p_dev_working, length, p_dev_input2);

	cudaEventRecord(p_compute_handle->stop_excluding_copy, stream);
	cudaEventRecord(p_compute_handle->stop_including_copy, stream);

	cudaEventSynchronize(p_compute_handle->stop_including_copy);

	return 0;
}


int GPUSAXPYAsynchronousCopyHostToDevice(CUDAHandle handle, 
	int length, float A, float *p_input1, float *p_input2)
{
	UNUSED(A);

	GPUComputeHandle *p_compute_handle;
	p_compute_handle = (GPUComputeHandle*)handle;

	if (NULL == p_compute_handle)
		return -1;

	float *p_host_input1 = p_input1;
	float *p_host_input2 = p_input2;

	cudaStream_t stream = p_compute_handle->stream;
	float *p_dev_input1 = p_compute_handle->p_dev_input1;
	float *p_dev_input2 = p_compute_handle->p_dev_input2;
	float *p_dev_working = p_compute_handle->p_dev_working;

	cudaEventRecord(p_compute_handle->start_including_copy, stream);

	cudaMemcpyAsync(p_dev_input1, p_host_input1, length * sizeof(float),
		cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(p_dev_input2, p_host_input2, length * sizeof(float),
		cudaMemcpyHostToDevice, stream);
	return 0;
}

/**********************************************************************/

int GPUSAXPYAsynchronousCompute(CUDAHandle handle, 
	int length, float A, float *p_input1, float *p_input2)
{
	GPUComputeHandle *p_compute_handle;
	p_compute_handle = (GPUComputeHandle*)handle;

	if (NULL == p_compute_handle)
		return -1;

	float *p_host_input1 = p_input1;
	float *p_host_input2 = p_input2;

	cudaStream_t stream = p_compute_handle->stream;
	float *p_dev_input1 = p_compute_handle->p_dev_input1;
	float *p_dev_input2 = p_compute_handle->p_dev_input2;
	float *p_dev_working = p_compute_handle->p_dev_working;

	cudaEventRecord(p_compute_handle->start_excluding_copy, stream);

#ifdef _BEST_BLOCK_NUMBER
	int block_num;
	int threads_in_block = THREADS_IN_BLOCK ;
	block_num = (length + (threads_in_block - 1)) / threads_in_block;
	block_num *= threads_in_block;
#else
	int block_num = NUM_BLOCKS;
#endif

	KernelMultipleByConstant << < THREADS_IN_BLOCK, block_num, 0, stream >> > (
		p_dev_input1, A, length, p_dev_working);

	KernelAdd << < THREADS_IN_BLOCK, block_num, 0, stream >> > (
		p_dev_input2, p_dev_working, length, p_dev_input2);

	cudaEventRecord(p_compute_handle->stop_excluding_copy, stream);
	return 0;
}

/**********************************************************************/

int GPUSAXPYAsynchronousCopyDeviceToHost(CUDAHandle handle, 
	int length, float A, float *p_input1, float *p_input2)
{
	UNUSED(A);

	GPUComputeHandle *p_compute_handle;
	p_compute_handle = (GPUComputeHandle*)handle;

	if (NULL == p_compute_handle)
		return -1;

	float *p_host_input1 = p_input1;
	float *p_host_input2 = p_input2;

	cudaStream_t stream = p_compute_handle->stream;
	float *p_dev_input1 = p_compute_handle->p_dev_input1;
	float *p_dev_input2 = p_compute_handle->p_dev_input2;
	float *p_dev_working = p_compute_handle->p_dev_working;


	cudaMemcpyAsync(p_host_input2, p_dev_input2, length * sizeof(float),
		cudaMemcpyDeviceToHost, stream);

	cudaEventRecord(p_compute_handle->stop_including_copy, stream);

	return 0;
}

/**********************************************************************/

int GPUSAXPYAsynchronous(CUDAHandle handle, 
	int length, float A, float *p_input1, float *p_input2)
{
	GPUComputeHandle *p_compute_handle;
	p_compute_handle = (GPUComputeHandle*)handle;

	if (NULL == p_compute_handle)
		return -1;

	GPUSAXPYAsynchronousCopyHostToDevice(handle, 
		length, A, p_input1, p_input2);
	GPUSAXPYAsynchronousCompute(handle,
		length, A, p_input1, p_input2);
	GPUSAXPYAsynchronousCopyDeviceToHost(handle, 
		length, A, p_input1, p_input2);

	return 0;
}

/**********************************************************************/

int WaitComputingDone(CUDAHandle handle)
{
	GPUComputeHandle *p_compute_handle;
	p_compute_handle = (GPUComputeHandle*)handle;

	if (NULL == p_compute_handle)
		return -1;

	cudaEventSynchronize(p_compute_handle->stop_including_copy);
	
	return 0;
}

/**********************************************************************/

bool IsComputeDone(CUDAHandle handle)
{
	GPUComputeHandle *p_compute_handle;
	p_compute_handle = (GPUComputeHandle*)handle;

	if (NULL == p_compute_handle)
		return false;

	cudaError_t err;
	err = cudaEventQuery(p_compute_handle->stop_including_copy);

	if (cudaSuccess == err)
		return true;

	return false;
}

/**********************************************************************/

static __global__ void KernelWarmUp(void) 
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid;
}

/**********************************************************************/

CUDAHandle InitGPUCompute(int length)
{
	GPUComputeHandle *p_compute_handle;
	p_compute_handle = (GPUComputeHandle*)malloc(sizeof(GPUComputeHandle));

	cudaError_t err;
	err = cudaStreamCreate(&p_compute_handle->stream);

	cudaMalloc(&p_compute_handle->p_dev_input1, length * sizeof(float));
	cudaMalloc(&p_compute_handle->p_dev_input2, length * sizeof(float));
	cudaMalloc(&p_compute_handle->p_dev_working, length * sizeof(float));

	cudaEventCreate(&p_compute_handle->start_including_copy);
	cudaEventCreate(&p_compute_handle->stop_including_copy);
	cudaEventCreate(&p_compute_handle->start_excluding_copy);
	cudaEventCreate(&p_compute_handle->stop_excluding_copy);

	KernelWarmUp << < THREADS_IN_BLOCK, NUM_BLOCKS, 
		0, p_compute_handle->stream >> >();

	cudaDeviceSynchronize();
	return p_compute_handle;
}

/**********************************************************************/

void CloseGPUCompute(CUDAHandle handle)
{
	GPUComputeHandle *p_compute_handle; 
	p_compute_handle = (GPUComputeHandle*)handle;

	if (NULL == p_compute_handle)
		return;

	cudaEventDestroy(p_compute_handle->start_including_copy);
	cudaEventDestroy(p_compute_handle->stop_including_copy);
	cudaEventDestroy(p_compute_handle->start_excluding_copy);
	cudaEventDestroy(p_compute_handle->stop_excluding_copy);

	cudaError_t err;
	err = cudaStreamDestroy(p_compute_handle->stream);

	cudaFree(&p_compute_handle->p_dev_input1);
	cudaFree(&p_compute_handle->p_dev_input2);
	cudaFree(&p_compute_handle->p_dev_working);	


	free(p_compute_handle);
}

/**********************************************************************/

void GetElaspedTime(CUDAHandle handle, float *p_elasped_time_including_copy_in_ms,
	float *p_elasped_time_excluding_copy_in_ms)
{
	GPUComputeHandle *p_compute_handle;
	p_compute_handle = (GPUComputeHandle*)handle;

	if (NULL == p_compute_handle)
		return;

	
	float elasped_time_including_copy_in_ms;
	cudaEventElapsedTime(&elasped_time_including_copy_in_ms,
		p_compute_handle->start_including_copy,
		p_compute_handle->stop_including_copy);

	float elasped_time_excluding_copy_in_ms;
	cudaEventElapsedTime(&elasped_time_excluding_copy_in_ms,
		p_compute_handle->start_excluding_copy,
		p_compute_handle->stop_excluding_copy);


	*p_elasped_time_including_copy_in_ms = elasped_time_including_copy_in_ms;
	*p_elasped_time_excluding_copy_in_ms = elasped_time_excluding_copy_in_ms;
}

/**********************************************************************/


