#include <stdio.h>
#include <stdlib.h>

#include "common.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "separable_convolution_31_1400x1400_gpu.h"

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
#if(1)
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
#endif

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



LOCAL __constant__ float kernel_const_mem[1024];

#define COLUMN_CP_STEPS						\
	(KERNEL_LENGTH + 2*Y_NUM_THREADS - 1 + (Y_NUM_THREADS - 1))/(Y_NUM_THREADS)


template<int jj> __device__ int CopyToSharedMemColumn(
	int i, int j,
	int block_height, int extended_width, int shared_mem_pitch,
	int kernel_radius,
	float *p_input_in_block, float const *p_extended_input_dev)
{
	int jjj;
	jjj = COLUMN_CP_STEPS - jj;

	p_input_in_block[(threadIdx.y + jjj*blockDim.y)* shared_mem_pitch
		+ threadIdx.x]
		= p_extended_input_dev
		[(j + jjj*blockDim.y)*extended_width
		+ kernel_radius + i];

	p_input_in_block[(threadIdx.y + jjj*blockDim.y)* shared_mem_pitch
		+ threadIdx.x + blockDim.x]
		= p_extended_input_dev
		[(j + jjj*blockDim.y)*extended_width
		+ kernel_radius + i + blockDim.x];


	CopyToSharedMemColumn<jj - 1>(i, j,
		block_height, extended_width, shared_mem_pitch, kernel_radius,
		p_input_in_block, p_extended_input_dev);
}/*CopyToSharedMemRow*/

template<> __device__ int CopyToSharedMemColumn<1>(
	int i, int j,
	int block_height, int extended_width, int shared_mem_pitch,
	int kernel_radius,
	float *p_input_in_block, float const *p_extended_input_dev)
{
	int jjj;
	jjj = COLUMN_CP_STEPS - 1;
	if (threadIdx.y + jjj * blockDim.y < block_height) {
		p_input_in_block[(threadIdx.y + jjj*blockDim.y)* shared_mem_pitch
			+ threadIdx.x]
			= p_extended_input_dev
			[(j + jjj*blockDim.y)*extended_width
			+ kernel_radius + i];

		p_input_in_block[(threadIdx.y + jjj*blockDim.y)* shared_mem_pitch
			+ threadIdx.x + blockDim.x]
			= p_extended_input_dev
			[(j + jjj*blockDim.y)*extended_width
			+ kernel_radius + i + blockDim.x];
	}/*if */

	return 0;
}/*CopyToSharedMemRow<1>*/


LOCAL __global__ void SeparateConvolutionColumnGPU_31_1400x1400CU(
	int width, int height, float const *p_extended_input_dev,
	int kernel_length, float const *p_kernel_column_dev,
	float *p_column_done_extended_output_dev, const int padding)
{
	int i, j;
	int kernel_radius;
	int extended_width;

	extern __shared__ float shared_mem[];
	float *p_input_in_block;
	int block_height;
	int shared_mem_pitch;

	(void)p_kernel_column_dev;

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;

	p_input_in_block = &shared_mem[0];

	block_height = kernel_length + ( 2*blockDim.y - 1);

	shared_mem_pitch = 2 * blockDim.x;
	shared_mem_pitch += padding;

	j = 2 * blockDim.y*blockIdx.y + threadIdx.y;
	i = 2 * blockDim.x*blockIdx.x + threadIdx.x;

	int jj;
	float sum;

#if(0)
	jj = 0;

	do {
		p_input_in_block[(threadIdx.y + jj*blockDim.y)* shared_mem_pitch
			+ threadIdx.x]
			= p_extended_input_dev
			[(j + jj*blockDim.y)*extended_width
			+ kernel_radius + i];

		p_input_in_block[(threadIdx.y + jj*blockDim.y)* shared_mem_pitch
			+ threadIdx.x + blockDim.x]
			= p_extended_input_dev
			[(j + jj*blockDim.y)*extended_width
			+ kernel_radius + i + blockDim.x];
		jj++;
	} while (threadIdx.y + jj * blockDim.y < block_height);
	
#else
#if(0)
#pragma unroll (COLUMN_CP_STEPS - 1)
	for (int jj = 0; jj < (COLUMN_CP_STEPS - 1) ; jj++) {
			p_input_in_block[(threadIdx.y + jj*blockDim.y)* shared_mem_pitch
				+ threadIdx.x]
				= p_extended_input_dev
				[(j + jj*blockDim.y)*extended_width
				+ kernel_radius + i];

			p_input_in_block[(threadIdx.y + jj*blockDim.y)* shared_mem_pitch
				+ threadIdx.x + blockDim.x]
				= p_extended_input_dev
				[(j + jj*blockDim.y)*extended_width
				+ kernel_radius + i + blockDim.x];		
	}/*for */

	jj = (COLUMN_CP_STEPS - 1);
	if (threadIdx.y + jj * blockDim.y < block_height) {
		p_input_in_block[(threadIdx.y + jj*blockDim.y)* shared_mem_pitch
			+ threadIdx.x]
			= p_extended_input_dev
			[(j + jj*blockDim.y)*extended_width
			+ kernel_radius + i];

		p_input_in_block[(threadIdx.y + jj*blockDim.y)* shared_mem_pitch
			+ threadIdx.x + blockDim.x]
			= p_extended_input_dev
			[(j + jj*blockDim.y)*extended_width
			+ kernel_radius + i + blockDim.x];
	}
#else
	CopyToSharedMemColumn<COLUMN_CP_STEPS>(i, j, block_height, extended_width, shared_mem_pitch,
		kernel_radius, p_input_in_block, p_extended_input_dev);
#endif
#endif
	__syncthreads();

	sum = 0;
#pragma unroll KERNEL_LENGTH
	for (jj = 0; jj < KERNEL_LENGTH; jj++) {
		sum += kernel_const_mem[jj] * p_input_in_block[
			(threadIdx.y + jj)*shared_mem_pitch + threadIdx.x];
	}/*for kernel*/

	p_column_done_extended_output_dev[j*extended_width + kernel_radius + i]
		= sum;


	sum = 0;
#pragma unroll KERNEL_LENGTH
	for (jj = 0; jj < KERNEL_LENGTH; jj++) {
		sum += kernel_const_mem[jj] * p_input_in_block[
			(threadIdx.y + jj)*shared_mem_pitch + threadIdx.x + blockDim.x];
	}/*for kernel*/

	p_column_done_extended_output_dev[j*extended_width + kernel_radius + i + blockDim.x]
		= sum;


	sum = 0;
#pragma unroll KERNEL_LENGTH
	for (jj = 0; jj < KERNEL_LENGTH; jj++) {
		sum += kernel_const_mem[jj] * p_input_in_block[
			(threadIdx.y + jj + blockDim.y)*shared_mem_pitch + threadIdx.x];
	}/*for kernel*/

	p_column_done_extended_output_dev[(j + blockDim.y)*extended_width + kernel_radius + i]
		= sum;


	sum = 0;
#pragma unroll KERNEL_LENGTH
	for (jj = 0; jj < KERNEL_LENGTH; jj++) {
		sum += kernel_const_mem[jj] * p_input_in_block[
			(threadIdx.y + jj + blockDim.y)*shared_mem_pitch + threadIdx.x + blockDim.x];
	}/*for kernel*/

	p_column_done_extended_output_dev[(j + blockDim.y)*extended_width + kernel_radius + i + blockDim.x]
		= sum;

}/*SeparateConvolutionColumnGPU_31_1400x1400CU*/

#define ROW_CP_STEPS						\
	(KERNEL_LENGTH + 2*X_NUM_THREADS - 1 + (X_NUM_THREADS - 1))/(X_NUM_THREADS)


template<int ii> __device__ int CopyToSharedMemRow(
	int i, int j,
	int block_width, int extended_width, int shared_mem_pitch,
	float *p_input_in_block, float const *p_column_done_extended_input_dev)
{
	int iii;

	iii = ROW_CP_STEPS - ii;

	CopyToSharedMemRow<ii - 1>(i, j,
		block_width, extended_width, shared_mem_pitch,
		p_input_in_block, p_column_done_extended_input_dev);

	p_input_in_block[threadIdx.y*shared_mem_pitch
		+ iii*blockDim.x + threadIdx.x] =
		p_column_done_extended_input_dev[j*extended_width
		+ iii*blockDim.x + i];

	p_input_in_block[(threadIdx.y + blockDim.y)*shared_mem_pitch
		+ iii*blockDim.x + threadIdx.x] =
		p_column_done_extended_input_dev[(j + blockDim.y)*extended_width
		+ iii*blockDim.x + i];

}/*CopyToSharedMemRow*/

template<> __device__ int CopyToSharedMemRow<1>(
	int i, int j,
	int block_width, int extended_width, int shared_mem_pitch,
	float *p_input_in_block, float const *p_column_done_extended_input_dev)
{
	int iii = ROW_CP_STEPS - 1;

	if (threadIdx.x + iii * blockDim.x < block_width) {
		p_input_in_block[threadIdx.y*shared_mem_pitch
			+ iii*blockDim.x + threadIdx.x] =
			p_column_done_extended_input_dev[j*extended_width
			+ iii*blockDim.x + i];

		p_input_in_block[(threadIdx.y + blockDim.y)*shared_mem_pitch
			+ iii*blockDim.x + threadIdx.x] =
			p_column_done_extended_input_dev[(j + blockDim.y)*extended_width
			+ iii*blockDim.x + i];
	}/*if */

	return 0;
}/*CopyToSharedMemRow<1>*/


LOCAL __global__ void SeparateConvolutionRowGPU_31_1400x1400CU(
	int width, int height, float const *p_column_done_extended_input_dev,
	int kernel_length, float const *p_kernel_row_dev,
	float *p_output_dev, const int padding)
{
	int i, j;
	int kernel_radius;
	int extended_width;

	extern __shared__ float shared_mem[];
	float *p_input_in_block;
	int block_width;
	int shared_mem_pitch;


	(void)p_kernel_row_dev;

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;

	p_input_in_block = &shared_mem[0];
	block_width = kernel_length + (2 * blockDim.x - 1);

	shared_mem_pitch = block_width;	
	shared_mem_pitch += padding;


	j = 2 * blockDim.y*blockIdx.y + threadIdx.y;
	i = 2 * blockDim.x*blockIdx.x + threadIdx.x;	
	

	int ii;
	float sum;

	sum = 0;
	
#if(0)
	ii = 0;
	do {
		p_input_in_block[threadIdx.y*shared_mem_pitch
			+ ii*blockDim.x + threadIdx.x] =
			p_column_done_extended_input_dev[j*extended_width
			+ ii*blockDim.x + i];		
		
		p_input_in_block[(threadIdx.y + blockDim.y)*shared_mem_pitch
			+ ii*blockDim.x + threadIdx.x] =
			p_column_done_extended_input_dev[(j + blockDim.y)*extended_width
			+ ii*blockDim.x + i];

		ii++;
	} while (threadIdx.x + ii * blockDim.x < block_width);
	
#else


#if(1)

#pragma unroll (ROW_CP_STEPS - 1)
	for (int ii = 0; ii < (ROW_CP_STEPS - 1); ii++) {
		if (threadIdx.x + ii * blockDim.x < block_width) {
			p_input_in_block[threadIdx.y*shared_mem_pitch
				+ ii*blockDim.x + threadIdx.x] =
				p_column_done_extended_input_dev[j*extended_width
				+ ii*blockDim.x + i];

			p_input_in_block[(threadIdx.y + blockDim.y)*shared_mem_pitch
				+ ii*blockDim.x + threadIdx.x] =
				p_column_done_extended_input_dev[(j + blockDim.y)*extended_width
				+ ii*blockDim.x + i];
		}
	}/*for */

	ii = ROW_CP_STEPS - 1;
	if (threadIdx.x + ii * blockDim.x < block_width) {
		p_input_in_block[threadIdx.y*shared_mem_pitch
			+ ii*blockDim.x + threadIdx.x] =
			p_column_done_extended_input_dev[j*extended_width
			+ ii*blockDim.x + i];

		p_input_in_block[(threadIdx.y + blockDim.y)*shared_mem_pitch
			+ ii*blockDim.x + threadIdx.x] =
			p_column_done_extended_input_dev[(j + blockDim.y)*extended_width
			+ ii*blockDim.x + i];
	}

#else
	CopyToSharedMemRow<ROW_CP_STEPS>(i, j,
		block_width, extended_width, shared_mem_pitch,
		p_input_in_block, p_column_done_extended_input_dev);
	

#endif
#endif
	__syncthreads();

	sum = 0;
#pragma unroll KERNEL_LENGTH
	for (ii = 0; ii < KERNEL_LENGTH; ii++) {
		sum += kernel_const_mem[ii] * p_input_in_block[
			threadIdx.y*shared_mem_pitch + ii + threadIdx.x];
	}/*for kernel_length*/

	p_output_dev[j*width + i] = sum;

	sum = 0;
#pragma unroll KERNEL_LENGTH
	for (ii = 0; ii < KERNEL_LENGTH; ii++) {
		sum += kernel_const_mem[ii] * p_input_in_block[
			threadIdx.y*shared_mem_pitch + ii + threadIdx.x + blockDim.x];
	}/*for kernel_length*/

	p_output_dev[j*width + i + blockDim.x] = sum;

	sum = 0;
#pragma unroll KERNEL_LENGTH
	for (ii = 0; ii < KERNEL_LENGTH; ii++) {
		sum += kernel_const_mem[ii] * p_input_in_block[
			(threadIdx.y + blockDim.y)*shared_mem_pitch + ii + threadIdx.x];
	}/*for kernel_length*/

	p_output_dev[(j + blockDim.y)*width + i] = sum;

	sum = 0;
#pragma unroll KERNEL_LENGTH
	for (ii = 0; ii < KERNEL_LENGTH; ii++) {
		sum += kernel_const_mem[ii] * p_input_in_block[
			(threadIdx.y + blockDim.y)*shared_mem_pitch + ii + threadIdx.x + blockDim.x];
	}/*for kernel_length*/

	
	p_output_dev[(j + blockDim.y)* width + i + blockDim.x] = sum;
}/*SeparateConvolutionRowGPU_31_1400x1400CU*/

#define WARP_SIZE					(32)

int SeparableConvolutionColumnGPU_31_1400x1400(
	dim3 num_blocks, dim3 num_threads,
	int width, int height, float const *p_extended_input_dev,
	int kernel_length, float const *p_kernel_column_host,
	float *p_column_done_extended_output_dev)
{
	int extended_width;
	float *p_kernel_const_dev;
	int shared_mem_size;
	int kernel_radius;
	
	int block_height;
	int padding;


	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
		return -2;

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;

	block_height = kernel_length + (num_threads.y - 1);
	block_height += num_threads.y;

/*
	padding
	= WARP_SIZE*n - (block_size + num_threads + (WARP_SIZE - num_threads))
*/

/*
	padding = num_threads.x + (WARP_SIZE - num_threads.x);
*/
	padding = 0;
	shared_mem_size = sizeof(float)
		* ( 2 * num_threads.x + padding) *( block_height);	

	HANDLE_ERROR(cudaGetSymbolAddress((void **)&p_kernel_const_dev,
		kernel_const_mem));

	HANDLE_ERROR(cudaMemcpy(p_kernel_const_dev, p_kernel_column_host,
		kernel_length * sizeof(float), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemset(p_column_done_extended_output_dev, 0,
		extended_width*height * sizeof(float)));

	num_blocks.x /= 2;
	num_blocks.y /= 2;
	
	SeparateConvolutionColumnGPU_31_1400x1400CU
		<< <num_blocks, num_threads, shared_mem_size >> >
		(width, height, p_extended_input_dev, kernel_length,
			NULL, p_column_done_extended_output_dev, padding);

	getLastCudaError("SeparateConvolutionColumnGPU_31_1400x1400CU");
	return 0;
}/*SeparableConvolutionColumnGPU_31_1400x1400*/


int SeparableConvolutionRowGPU_31_1400x1400(
	dim3 num_blocks, dim3 num_threads,
	int width, int height, float const *p_column_done_extended_input_dev,
	int kernel_length, float const *p_kernel_row_host,
	float *p_output_dev)
{
	float *p_kernel_const_dev;
	int shared_mem_size;
	int kernel_radius;

	int block_width;	
	int padding;

	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
		return -2;

	kernel_radius = kernel_length / 2;
	block_width = kernel_length + (num_threads.x - 1);
	block_width += num_threads.x;
	//int n;
	//n = (block_width + (num_threads.x - 1)) / num_threads.x;

	//block_width = num_threads.x * n;
/*
	padding 
	= WARP_SIZE*n - (block_size + (WARP_SIZE - num_threads))
*/
	{
		int temp = block_width + (WARP_SIZE - num_threads.x);

		padding = WARP_SIZE*((temp + (WARP_SIZE - 1)) / WARP_SIZE)
			- temp;
	}/*local variable*/

	shared_mem_size = sizeof(float) 
		* (block_width + padding) *(2*num_threads.y);
	
	HANDLE_ERROR(cudaGetSymbolAddress((void **)&p_kernel_const_dev,
		kernel_const_mem));

	HANDLE_ERROR(cudaMemcpy(p_kernel_const_dev, p_kernel_row_host,
		kernel_length * sizeof(float), cudaMemcpyHostToDevice));

	num_blocks.x /= 2;
	num_blocks.y /= 2;

	SeparateConvolutionRowGPU_31_1400x1400CU
		<< <num_blocks, num_threads, shared_mem_size >> >
		(width, height, p_column_done_extended_input_dev,
			kernel_length, NULL, p_output_dev, padding);

	getLastCudaError("SeparableConvolutionRowGPU_31_1400x1400");

	return 0;
}/*SeparableConvolutionRowGPU_31_1400x1400*/
