#include <stdio.h>
#include <stdlib.h>

#include "common.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "separable_convolution_gpu.h"

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


LOCAL __global__ void SeparateConvolutionRowGPULinearMemoryCU(
	int width, int height, float const *p_extended_input_dev,
	int kernel_length, float const *p_kernel_row_dev,
	float *p_row_done_extended_output_dev)
{
	int i, j;
	int kernel_radius;
	int extended_width;

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;

	j = blockDim.y*blockIdx.y + threadIdx.y;
	for (; j < height; j += blockDim.y * gridDim.y) {
		

		i = blockDim.x*blockIdx.x + threadIdx.x;
		for (; i < width; i += blockDim.x * gridDim.x) {

			int jj;
			int x, y_mul_input_width;
			float sum;

			sum = 0;
			x = kernel_radius + i;
			y_mul_input_width = j*extended_width;
			
			for (jj = 0; jj < kernel_length; jj++) {

				sum += p_kernel_row_dev[jj]
					* p_extended_input_dev[y_mul_input_width + x];

				y_mul_input_width += extended_width;
			}/*for kernel*/

			p_row_done_extended_output_dev[j*extended_width + x]
				= sum;
		}/*for width*/

	}/*for j*/

}/*SeparateConvolutionRowGPULinearMemoryCU*/


LOCAL __global__ void SeparateConvolutionColumnGPULinearMemoryCU(
	int width, int height, float const *p_row_done_extended_input_dev,
	int kernel_length, float const *p_kernel_column_dev,
	float *p_output_dev)
{

	int i, j;	
	int extended_width;

	extended_width = width + kernel_length - 1;


	j = blockDim.y*blockIdx.y + threadIdx.y;
	for (; j < height; j += blockDim.y * gridDim.y) {

		i = blockDim.x*blockIdx.x + threadIdx.x;
		for (; i < width; i += blockDim.x * gridDim.x) {

			int x, y_mul_input_width;
			int ii;
			float sum;

			sum = 0;
			y_mul_input_width = j*extended_width;
			x = i;

			for (ii = 0; ii < kernel_length; ii++) {

				sum += p_kernel_column_dev[ii]
					* p_row_done_extended_input_dev[y_mul_input_width + x];
				x += 1;
			}/*for kernel_length*/

			p_output_dev[j*width + i] = sum;
		}/*for width*/

	}/*for j*/

}/*SeparateConvolutionColumnGPULinearMemoryCU*/


int SeparableConvolutionRowGPULinearMemory(
	dim3 num_blocks, dim3 num_threads,
	int width, int height, float const *p_extended_input_dev,
	int kernel_length, float const *p_kernel_row_dev,
	float *p_row_done_extended_output_dev)
{
	int extended_width;

	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
		return -2;

	extended_width = width + kernel_length - 1;

	
	HANDLE_ERROR(cudaMemset(p_row_done_extended_output_dev, 0,
		extended_width*height*sizeof(float)));

	SeparateConvolutionRowGPULinearMemoryCU << <num_blocks, num_threads >> >
		(width, height, p_extended_input_dev, kernel_length,
			p_kernel_row_dev, p_row_done_extended_output_dev);

	getLastCudaError("SeparateConvolutionRowGPULinearMemoryCU");
	return 0;
}/*SeparateConvolutionRowGPULinearMemory*/


int SeparableConvolutionColumnGPULinearMemory(
	dim3 num_blocks, dim3 num_threads,
	int width, int height, float const *p_row_done_extended_input_dev,
	int kernel_length, float const *p_kernel_column_dev,
	float *p_output_dev)
{
	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
		return -2;

	SeparateConvolutionColumnGPULinearMemoryCU << <num_blocks, num_threads >> >
		(width, height, p_row_done_extended_input_dev,
			kernel_length, p_kernel_column_dev, p_output_dev);

	getLastCudaError("SeparateConvolutionColumnGPULinearMemoryCU");
	return 0;
}/*SeparateConvolutionColumnGPULinearMemory*/




__constant__ float kernel_const_mem[1024];


LOCAL __global__ void SeparateConvolutionRowGPUKernelInConstCU(
	int width, int height, float const *p_extended_input_dev,
	int kernel_length, float const *p_kernel_row_dev,
	float *p_row_done_extended_output_dev)
{
	int i, j;
	int kernel_radius;
	int extended_width;

	(void) p_kernel_row_dev;

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;

	j = blockDim.y*blockIdx.y + threadIdx.y;
	for (; j < height; j += blockDim.y * gridDim.y) {

		i = blockDim.x*blockIdx.x + threadIdx.x;
		for (; i < width; i += blockDim.x * gridDim.x) {

			int jj;
			int x, y_mul_input_width;
			float sum;

			sum = 0;
			x = kernel_radius + i;
			y_mul_input_width = j*extended_width;

			for (jj = 0; jj < kernel_length; jj++) {

				sum += kernel_const_mem[jj]
					* p_extended_input_dev[y_mul_input_width + x];

				y_mul_input_width += extended_width;
			}/*for kernel*/

			p_row_done_extended_output_dev[j*extended_width + x]
				= sum;
		}/*for width*/

	}/*for j*/

}/*SeparateConvolutionRowGPUKernelInConstCU*/


LOCAL __global__ void SeparateConvolutionColumnGPUKernelInConstCU(
	int width, int height, float const *p_row_done_extended_input_dev,
	int kernel_length, float const *p_kernel_column_dev,
	float *p_output_dev)
{

	int i, j;
	int extended_width;

	(void)p_kernel_column_dev;

	extended_width = width + kernel_length - 1;


	j = blockDim.y*blockIdx.y + threadIdx.y;
	for (; j < height; j += blockDim.y * gridDim.y) {

		i = blockDim.x*blockIdx.x + threadIdx.x;
		for (; i < width; i += blockDim.x * gridDim.x) {

			int x, y_mul_input_width;
			int ii;
			float sum;

			sum = 0;
			y_mul_input_width = j*extended_width;
			x = i;

			for (ii = 0; ii < kernel_length; ii++) {
				sum += kernel_const_mem[ii]
					* p_row_done_extended_input_dev[y_mul_input_width + x];

				x += 1;
			}/*for kernel_length*/

			p_output_dev[j*width + i] = sum;
		}/*for width*/

	}/*for j*/

}/*SeparateConvolutionColumnGPUKernelInConstCU*/


int SeparableConvolutionRowGPUKernelInConst(
	dim3 num_blocks, dim3 num_threads,
	int width, int height, float const *p_extended_input_dev,
	int kernel_length, float const *p_kernel_row_host,
	float *p_row_done_extended_output_dev)
{
	int extended_width;
	float *p_kernel_const_dev;

	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
		return -2;

	extended_width = width + kernel_length - 1;

	HANDLE_ERROR(cudaGetSymbolAddress((void **)&p_kernel_const_dev,
		kernel_const_mem));

	HANDLE_ERROR(cudaMemcpy(p_kernel_const_dev, p_kernel_row_host,
		kernel_length * sizeof(float), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemset(p_row_done_extended_output_dev, 0,
		extended_width*height * sizeof(float)));

	SeparateConvolutionRowGPUKernelInConstCU << <num_blocks, num_threads >> >
		(width, height, p_extended_input_dev, kernel_length,
			NULL, p_row_done_extended_output_dev);

	getLastCudaError("SeparateConvolutionRowGPUKernelInConstCU");
	return 0;
}/*SeparableConvolutionRowGPUKernelInConst*/


int SeparableConvolutionColumnGPUKernelInConst(
	dim3 num_blocks, dim3 num_threads,
	int width, int height, float const *p_row_done_extended_input_dev,
	int kernel_length, float const *p_kernel_column_host,
	float *p_output_dev)
{
	float *p_kernel_const_dev;

	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
		return -2;


	HANDLE_ERROR(cudaGetSymbolAddress((void **)&p_kernel_const_dev,
		kernel_const_mem));

	HANDLE_ERROR(cudaMemcpy(p_kernel_const_dev, p_kernel_column_host,
		kernel_length * sizeof(float), cudaMemcpyHostToDevice));

	SeparateConvolutionColumnGPUKernelInConstCU << <num_blocks, num_threads >> >
		(width, height, p_row_done_extended_input_dev,
			kernel_length, NULL, p_output_dev);

	getLastCudaError("SeparateConvolutionColumnGPUKernelInConstCU");
	return 0;
}/*SeparableConvolutionColumnGPUKernelInConst*/



LOCAL __global__ void SeparateConvolutionRowGPUKernelInConstSharedMemCU(
	int width, int height, float const *p_extended_input_dev,
	int kernel_length, float const *p_kernel_row_dev,
	float *p_row_done_extended_output_dev)
{
	int i, j;
	int kernel_radius;
	int extended_width;

	extern __shared__ float shared_mem[];
	float *p_input_in_block;


	(void)p_kernel_row_dev;

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;

	p_input_in_block = &shared_mem[0];

#ifdef _ROW_DATA_IN_CONSECUTIVE_SHARED_MEN

	int input_in_block_height;

	input_in_block_height = blockDim.y + 2 * kernel_radius;

	j = blockDim.y*blockIdx.y + threadIdx.y;
	for (; j < height; j += blockDim.y * gridDim.y) {

		i = blockDim.x*blockIdx.x + threadIdx.x;
		for (; i < width; i += blockDim.x * gridDim.x) {

			int jj;
			int x;
			float sum;

			sum = 0;
			x = kernel_radius + i;
			jj = 0;
			do {
				if (threadIdx.y + jj*blockDim.y <
					blockDim.y + 2 * kernel_radius)
				{
					p_input_in_block[threadIdx.x * input_in_block_height
						+ jj*blockDim.y + threadIdx.y]
						= p_extended_input_dev
						[(j + jj*blockDim.y)*extended_width
						+ kernel_radius + i];
				}/*if */

				jj++;
			} while (jj * blockDim.y <  blockDim.y + 2 * kernel_radius);

			__syncthreads();

			for (jj = 0; jj < kernel_length; jj++) {
				sum += kernel_const_mem[jj] * p_input_in_block[
					threadIdx.x*input_in_block_height + jj + threadIdx.y];
			}/*for kernel*/

			p_row_done_extended_output_dev[j*extended_width + kernel_radius + i]
				= sum;

			__syncthreads();
		}/*for width*/

	}/*for j*/

#else

	j = blockDim.y*blockIdx.y + threadIdx.y;
	for (; j < height; j += blockDim.y * gridDim.y) {

		i = blockDim.x*blockIdx.x + threadIdx.x;
		for (; i < width; i += blockDim.x * gridDim.x) {

			int jj;
			int x;
			float sum;

			sum = 0;
			x = kernel_radius + i;

			jj = 0;
			do {
				if (threadIdx.y + jj*blockDim.y <  
					blockDim.y + 2 * kernel_radius)
				{
					p_input_in_block[(threadIdx.y + jj*blockDim.y)*blockDim.x
						+ threadIdx.x] 
						= p_extended_input_dev
						[ (j + jj*blockDim.y)*extended_width
						+ kernel_radius + i];
				}/*if */
				
				jj++;
			} while (jj * blockDim.y <  blockDim.y + 2 * kernel_radius);
		
			__syncthreads();


			for (jj = 0; jj < kernel_length; jj++) {
				sum += kernel_const_mem[jj]* p_input_in_block[
					(threadIdx.y + jj)*blockDim.x + threadIdx.x];
			}/*for kernel*/

			p_row_done_extended_output_dev[j*extended_width + kernel_radius + i]
				= sum;

			__syncthreads();
		}/*for width*/
		
	}/*for j*/

#endif

}/*SeparateConvolutionRowGPUKernelInConstSharedMemCU*/


LOCAL __global__ void SeparateConvolutionColumnGPUKernelInConstSharedMemCU(
	int width, int height, float const *p_row_done_extended_input_dev,
	int kernel_length, float const *p_kernel_column_dev,
	float *p_output_dev)
{
	int i, j;
	int kernel_radius;
	int extended_width;

	extern __shared__ float shared_mem[];
	float *p_input_in_block;
	int input_in_block_width;

	(void)p_kernel_column_dev;

	kernel_radius = kernel_length / 2;
	extended_width = width + kernel_length - 1;

	p_input_in_block = &shared_mem[0];
	input_in_block_width = blockDim.x + 2 * kernel_radius;


	j = blockDim.y*blockIdx.y + threadIdx.y;
	for (; j < height; j += blockDim.y * gridDim.y) {

		i = blockDim.x*blockIdx.x + threadIdx.x;
		for (; i < width; i += blockDim.x * gridDim.x) {
			int ii;
			float sum;

			sum = 0;
			ii = 0;

			do {
				if (threadIdx.x + ii*blockDim.x < input_in_block_width)
				{
					p_input_in_block[threadIdx.y*input_in_block_width
						+ ii*blockDim.x + threadIdx.x] =
						p_row_done_extended_input_dev[j*extended_width 
						+ ii*blockDim.x + i];
				}/*if */
				
				ii++;
			} while (ii* blockDim.x < input_in_block_width);

			__syncthreads();


			for (ii = 0; ii < kernel_length; ii++) {
				sum += kernel_const_mem[ii]* p_input_in_block[
					threadIdx.y*input_in_block_width + ii + threadIdx.x];
			}/*for kernel_length*/
			
			p_output_dev[j*width + i] = sum;
			__syncthreads();
		}/*for width*/

	}/*for j*/

}/*SeparateConvolutionColumnGPUKernelInConstSharedMemCU*/


int SeparableConvolutionRowGPUKernelInConstSharedMem(
	dim3 num_blocks, dim3 num_threads,
	int width, int height, float const *p_extended_input_dev,
	int kernel_length, float const *p_kernel_row_host,
	float *p_row_done_extended_output_dev)
{
	int extended_width;
	float *p_kernel_const_dev;
	int shared_mem_size;
	int kernel_radius;

	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
		return -2;

	extended_width = width + kernel_length - 1;
	kernel_radius = kernel_length / 2;

	shared_mem_size = sizeof(float)*
		(num_threads.y + 2 * kernel_radius)*(num_threads.x);

	HANDLE_ERROR(cudaGetSymbolAddress((void **)&p_kernel_const_dev,
		kernel_const_mem));

	HANDLE_ERROR(cudaMemcpy(p_kernel_const_dev, p_kernel_row_host,
		kernel_length * sizeof(float), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemset(p_row_done_extended_output_dev, 0,
		extended_width*height * sizeof(float)));

	SeparateConvolutionRowGPUKernelInConstSharedMemCU 
		<< <num_blocks, num_threads, shared_mem_size >> >
		(width, height, p_extended_input_dev, kernel_length,
			NULL, p_row_done_extended_output_dev);
	
	getLastCudaError("SeparateConvolutionRowGPUKernelInConstCU");
	return 0;
}/*SeparableConvolutionRowGPUKernelInConstSharedMem*/


int SeparableConvolutionColumnGPUKernelInConstSharedMem(
	dim3 num_blocks, dim3 num_threads,
	int width, int height, float const *p_row_done_extended_input_dev,
	int kernel_length, float const *p_kernel_column_host,
	float *p_output_dev)
{
	float *p_kernel_const_dev;
	int shared_mem_size;
	int kernel_radius;

	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
		return -2;

	kernel_radius = kernel_length / 2;

	shared_mem_size = sizeof(float)*
		(num_threads.x + 2 * kernel_radius)*(num_threads.y);

	HANDLE_ERROR(cudaGetSymbolAddress((void **)&p_kernel_const_dev,
		kernel_const_mem));

	HANDLE_ERROR(cudaMemcpy(p_kernel_const_dev, p_kernel_column_host,
		kernel_length * sizeof(float), cudaMemcpyHostToDevice));

	SeparateConvolutionColumnGPUKernelInConstSharedMemCU 
		<< <num_blocks, num_threads, shared_mem_size >> >
		(width, height, p_row_done_extended_input_dev,
			kernel_length, NULL, p_output_dev);

	getLastCudaError("SeparateConvolutionColumnGPUKernelInConstCU");

	return 0;
}/*SeparableConvolutionColumnGPUKernelInConstSharedMem*/

