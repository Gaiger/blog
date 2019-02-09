
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "common.h"

#include "select_cuda_device.h"
#include "cuda_runtime.h"


#include "separable_convolution_cpu.h"
#include "separable_convolution_gpu.h"

#ifdef _MSC_VER 

#include <windows.h>
//#include <mmsystem.h>


unsigned int GetTime(void)
{
	return (unsigned int)timeGetTime();
}/*GetTime*/

#endif

static void CudaHandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (CudaHandleError( err, __FILE__, __LINE__ ))



#define SAFE_FREE(PTR)				if(NULL != (PTR)) \
										free(PTR);	\
									(PTR) = NULL;


#define TIMER_BEGIN(TIMER_LABEL)	\
									{	unsigned int tBegin##TIMER_LABEL, tEnd##TIMER_LABEL; \
										tBegin##TIMER_LABEL = GetTime();

#define TIMER_END(TIMER_LABEL)		\
										tEnd##TIMER_LABEL = GetTime();\
										fprintf(stderr, "cost time = %d ms\n", \
										 tEnd##TIMER_LABEL - tBegin##TIMER_LABEL);	\
									}

#define TIMER_LOOP_BEGIN(TIMER_LABEL, ROUND)	 \
									{ \
										int ijk##TIMER_LABEL; \
										int repeat_times_##TIMER_LABEL = ROUND; \
										unsigned int tBegin##TIMER_LABEL, tEnd##TIMER_LABEL; \
										fprintf(stderr, "\r\n%s, ROUND = %d\r\n", \
										#TIMER_LABEL, repeat_times_##TIMER_LABEL); \
										tBegin##TIMER_LABEL = GetTime(); \
										for (ijk##TIMER_LABEL = 0; ijk##TIMER_LABEL < repeat_times_##TIMER_LABEL; (ijk##TIMER_LABEL)++) {


#define TIMER_LOOP_END(TIMER_LABEL)	 \
										} \
										tEnd##TIMER_LABEL = GetTime(); \
										fprintf(stderr, " %d ms (%.2f ms per round)\n\n", \
											tEnd##TIMER_LABEL - tBegin##TIMER_LABEL, \
											(tEnd##TIMER_LABEL - tBegin##TIMER_LABEL)/(float)repeat_times_##TIMER_LABEL); \
									}
									
void SeparableCudaMeasureRuntime(char *p_label_str, int rounds,
	int width, int height, float *p_extended_input_dev,
	int kernel_length, float  *p_kernel_row_dev, float *p_kernel_column_dev,
	float *p_row_done_extended_output_dev)
{
	if (NULL == p_label_str)
		return;
TIMER_LOOP_BEGIN(p_label_str, rounds)
	
TIMER_LOOP_END(p_label_str)
}/*SeparableCudaMeasureRuntime*/


int NumberOfThreadsCorrection(int width, int height, dim3 *p_num_threads)
{
	int current_running_device_id;
	int x_number_threads, y_number_threads;
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDevice(&current_running_device_id));

	cudaGetDeviceProperties(&prop, current_running_device_id);

	y_number_threads = 32;
	
	if (height > y_number_threads)
	{
		while (height != y_number_threads *(height / y_number_threads))
		{
			y_number_threads -= 1;
			if (1 == y_number_threads)
				break;
		}
	}/*uf */


	x_number_threads = prop.maxThreadsDim[0] / y_number_threads;
	//x_number_threads = 32;

	if (width > x_number_threads)
	{
		while (width != x_number_threads *( width / x_number_threads))
		{
			x_number_threads -= 1;
			if (1 == x_number_threads)
				break;
		}
	}/*if */

	p_num_threads->x = x_number_threads;
	p_num_threads->y = y_number_threads;
	p_num_threads->z = 1;	
	
	
	printf(" X threads = %d\r\n", p_num_threads->x);
	printf(" Y threads = %d\r\n", p_num_threads->y);
	return 0;
}/*NumberOfThreadsCorrection*/


int main(int argc, char *argv[])
{
	int i, j;
	int width, height;
	int extended_width, extended_height;

	int kernel_length, kernel_radius;

	float *p_kernel_row;
	float *p_kernel_column;

	float *p_input, *p_extended_input;
	float *p_output;
	float *p_separable_column_intermediate;
	float *p_separable_output_cpu;


	printf("input = %d x %d\r\n", WIDTH, HEIGHT);
	printf("kernel :: %d x %d\r\n",
		(2 * KERNEL_RADIUS + 1), (2 * KERNEL_RADIUS + 1));

	width = WIDTH;
	height = HEIGHT;
	kernel_radius = KERNEL_RADIUS;
	extended_width = width + 2 * kernel_radius;
	extended_height = height + 2 * kernel_radius;
	kernel_length = KERNEL_LENGTH;

#ifdef _HOST_PIN
	HANDLE_ERROR(cudaMallocHost((void**)&p_kernel_column,
		kernel_length * sizeof(float)));
	HANDLE_ERROR(cudaMallocHost((void**)&p_kernel_row,
		kernel_length * sizeof(float)));
#else
	p_kernel_column = (float*)malloc(kernel_length * sizeof(float));
	p_kernel_row = (float*)malloc(kernel_length * sizeof(float));	
#endif

	{
		float shift_value;
		for (i = 0; i < kernel_length; i++)
			p_kernel_column[i] = (float)i + 1;

		shift_value = p_kernel_column[kernel_length - 1];

		for (i = 0; i < kernel_length; i++)
			p_kernel_row[i] = (float)i + shift_value + 1;

	}/*local variable*/

	p_input = (float*)malloc(width*height * sizeof(float));
	for (j = 0; j < height; j++) {
		for (i = 0; i < width; i++) {
			p_input[j*width + i] = (float)(i + j*width + 1);
		}/*i*/
	}/*j*/

#ifdef _HOST_PIN
	HANDLE_ERROR(cudaMallocHost((void**)&p_extended_input,
		extended_width*extended_height * sizeof(float)));
#else
	p_extended_input = (float*)malloc(
		extended_width*extended_height * sizeof(float));
#endif

	for (j = 0; j < extended_height; j++) {
		for (i = 0; i < extended_width; i++) {

			if (i < kernel_radius || i >= (kernel_radius + width)
				|| j < kernel_radius || j >= (kernel_radius + height)
				)
			{
				p_extended_input[i + extended_width*j] = 0;
			}
			else
			{
				int ii, jj;
				jj = j - kernel_radius;
				ii = i - kernel_radius;

				p_extended_input[i + extended_width*j]
					= p_input[jj*width + ii];
			}/*if */

		}/*for j*/
	}/*for i*/

	p_output = (float*)malloc(
		width*height * sizeof(float));

	p_separable_column_intermediate = (float*)malloc(
		extended_width*height * sizeof(float));

	p_separable_output_cpu = (float*)malloc(
		width*height * sizeof(float));

#if(1)
TIMER_LOOP_BEGIN(SEPAREATE_CONVOLUTION_AVX, ROUND)
	SeparableConvolutionColumnAVX(width, height, p_extended_input,
			kernel_length, p_kernel_column, p_separable_column_intermediate);

	SeparableConvolutionRowAVX(width, height, p_separable_column_intermediate,
		kernel_length, p_kernel_row, p_separable_output_cpu);
TIMER_LOOP_END(SEPAREATE_CONVOLUTION_AVX)
#endif

	SelectCudaDevice();
	
	float *p_separable_output_gpu;

	float *p_kernel_column_dev, *p_kernel_row_dev;

	float *p_extended_input_dev;
	float *p_separable_column_intermediate_dev;
	float *p_separable_output_dev;
	
#ifdef _HOST_PIN
	HANDLE_ERROR(cudaMallocHost((void**)&p_separable_output_gpu,
		width*height * sizeof(float)));
#else
	p_separable_output_gpu = (float*)malloc(
		width*height * sizeof(float));
#endif

	HANDLE_ERROR(cudaMalloc((void**)&p_kernel_column_dev,
		kernel_length * sizeof(float)));

	HANDLE_ERROR(cudaMalloc((void**)&p_kernel_row_dev,
		kernel_length * sizeof(float)));


	HANDLE_ERROR(cudaMalloc((void**)&p_extended_input_dev,
		extended_width*extended_height * sizeof(float)));

	HANDLE_ERROR(cudaMalloc((void**)&p_separable_column_intermediate_dev,
		extended_width*height * sizeof(float)));

	HANDLE_ERROR(cudaMalloc((void**)&p_separable_output_dev,
		width*height * sizeof(float)));

TIMER_LOOP_BEGIN(CUDA_DUMMY, ROUND)
	HANDLE_ERROR(cudaMemcpy(p_extended_input_dev, p_extended_input,
		extended_width*extended_height * sizeof(float),
		cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpy(p_kernel_column_dev, p_kernel_row,
		kernel_length * sizeof(float),
		cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(p_kernel_row_dev, p_kernel_column,
		kernel_length * sizeof(float),
		cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpy(p_separable_output_gpu, p_separable_output_dev,
		width*height * sizeof(float),
		cudaMemcpyDeviceToHost));
TIMER_LOOP_END(CUDA_DUMMY)


#if(1)
TIMER_LOOP_BEGIN(SEPAREATE_CONVOLUTION_CUDA_LINEAR_MEM, ROUND)

	HANDLE_ERROR(cudaMemcpy(p_extended_input_dev, p_extended_input,
		extended_width*extended_height * sizeof(float),
		cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(p_kernel_column_dev, p_kernel_column,
		kernel_length * sizeof(float),
		cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(p_kernel_row_dev, p_kernel_row,
		kernel_length * sizeof(float),
		cudaMemcpyHostToDevice));
	dim3 num_blocks, num_threads;
	num_threads.x = X_NUM_THREADS;
	num_threads.y = Y_NUM_THREADS;
	num_blocks.x = (width + (num_threads.x - 1)) / num_threads.x;
	num_blocks.y = (height + (num_threads.y - 1)) / num_threads.y;

	SeparableConvolutionColumnGPULinearMemory(num_blocks, num_threads, width, height,
		p_extended_input_dev, kernel_length, p_kernel_column_dev,
		p_separable_column_intermediate_dev);

	SeparableConvolutionRowGPULinearMemory(num_blocks, num_threads, width, height,
		p_separable_column_intermediate_dev, kernel_length, p_kernel_row_dev,
		p_separable_output_dev);

	HANDLE_ERROR(cudaMemcpy(p_separable_output_gpu, p_separable_output_dev,
		width*height * sizeof(float),
		cudaMemcpyDeviceToHost));
TIMER_LOOP_END(SEPAREATE_CONVOLUTION_CUDA_LINEAR_MEM)
#endif


#if(1)
	TIMER_LOOP_BEGIN(SEPAREATE_CONVOLUTION_CUDA_KERNEL_IN_CONST, ROUND)

		HANDLE_ERROR(cudaMemcpy(p_extended_input_dev, p_extended_input,
			extended_width*extended_height * sizeof(float),
			cudaMemcpyHostToDevice));

	dim3 num_blocks, num_threads;
	num_threads.x = X_NUM_THREADS; num_threads.y = Y_NUM_THREADS;
	num_blocks.x = (width + (num_threads.x - 1)) / num_threads.x;
	num_blocks.y = (height + (num_threads.y - 1)) / num_threads.y;

	SeparableConvolutionColumnGPUKernelInConst(num_blocks, num_threads, width, height,
		p_extended_input_dev, kernel_length, p_kernel_column,
		p_separable_column_intermediate_dev);

	SeparableConvolutionRowGPUKernelInConst(num_blocks, num_threads, width, height,
		p_separable_column_intermediate_dev, kernel_length, p_kernel_row,
		p_separable_output_dev);

	HANDLE_ERROR(cudaMemcpy(p_separable_output_gpu, p_separable_output_dev,
		width*height * sizeof(float),
		cudaMemcpyDeviceToHost));
	TIMER_LOOP_END(SEPAREATE_CONVOLUTION_CUDA_KERNEL_IN_CONST)
#endif


#if(1)
	{
		dim3 num_blocks, num_threads;
		num_threads.x = X_NUM_THREADS; num_threads.y = Y_NUM_THREADS;
#ifndef _DEBUG
		NumberOfThreadsCorrection(width, height, &num_threads);
#endif
		num_blocks.x = (width + (num_threads.x - 1)) / num_threads.x;
		num_blocks.y = (height + (num_threads.y - 1)) / num_threads.y;

TIMER_LOOP_BEGIN(SEPAREATE_CONVOLUTION_CUDA_KERNEL_IN_CONST_SHARED_MEM, ROUND)

			HANDLE_ERROR(cudaMemcpy(p_extended_input_dev, p_extended_input,
				extended_width*extended_height * sizeof(float),
				cudaMemcpyHostToDevice));


		SeparableConvolutionColumnGPUKernelInConstSharedMem(num_blocks, num_threads,
			width, height,
			p_extended_input_dev, kernel_length, p_kernel_column,
			p_separable_column_intermediate_dev);

		SeparableConvolutionRowGPUKernelInConstSharedMem(num_blocks, num_threads,
			width, height,
			p_separable_column_intermediate_dev, kernel_length, p_kernel_row,
			p_separable_output_dev);

		HANDLE_ERROR(cudaMemcpy(p_separable_output_gpu, p_separable_output_dev,
			width*height * sizeof(float),
			cudaMemcpyDeviceToHost));

TIMER_LOOP_END(SEPAREATE_CONVOLUTION_CUDA_KERNEL_IN_CONST_SHARED_MEM)
	}/*local variable*/
#endif

#if(1)
	{
		dim3 num_blocks, num_threads;
		num_threads.x = X_NUM_THREADS; num_threads.y = Y_NUM_THREADS;
#ifndef _DEBUG
		NumberOfThreadsCorrection(width, height, &num_threads);
#endif
		num_blocks.x = (width + (num_threads.x - 1)) / num_threads.x;
		num_blocks.y = (height + (num_threads.y - 1)) / num_threads.y;

TIMER_LOOP_BEGIN(SEPAREATE_CONVOLUTION_CUDA_KERNEL_IN_CONST_SHARED_MEM_PADDING, ROUND)

		HANDLE_ERROR(cudaMemcpy(p_extended_input_dev, p_extended_input,
			extended_width*extended_height * sizeof(float),
			cudaMemcpyHostToDevice));


		SeparableConvolutionColumnGPUKernelInConstSharedMemPadding(num_blocks, num_threads,
			width, height,
			p_extended_input_dev, kernel_length, p_kernel_column,
			p_separable_column_intermediate_dev);

		SeparableConvolutionRowGPUKernelInConstSharedMemPadding(num_blocks, num_threads,
			width, height,
			p_separable_column_intermediate_dev, kernel_length, p_kernel_row,
			p_separable_output_dev);

		HANDLE_ERROR(cudaMemcpy(p_separable_output_gpu, p_separable_output_dev,
			width*height * sizeof(float),
			cudaMemcpyDeviceToHost));

TIMER_LOOP_END(SEPAREATE_CONVOLUTION_CUDA_KERNEL_IN_CONST_SHARED_MEM_PADDING)
	}/*local variable*/
#endif
#if(1)
	int count = 0;
	for (j = 0; j < height; j++) {
		for (i = 0; i < width; i++) {

			int is_over_tolerance;
			is_over_tolerance = 0;
			if (0 == p_separable_output_cpu[j*width + i])
			{
				if (0 != p_separable_output_gpu[j*width + i])
					is_over_tolerance = 1;
			}
			else
			{
				float tolerance;
				tolerance = p_separable_output_gpu[j*width + i] - p_separable_output_cpu[j*width + i];
				tolerance /= p_separable_output_cpu[j*width + i];
				tolerance = (float)fabs(tolerance);
				if (tolerance > 1.0e-6)
					is_over_tolerance = 1;
			}

			if (0 != is_over_tolerance)
			{
				printf("computs error : i = %d, j = %d"
					", value = %.f %.f\r\n", i, j, 
					p_separable_output_gpu[j*width + i], 
					p_separable_output_cpu[j*width + i]);
				count++;
				
			}

		}/*for i*/
	}/*for j*/

	if (0 <count)
		printf("error count = %d\r\n", count);
#endif

	HANDLE_ERROR(cudaFree(p_separable_output_dev));
	HANDLE_ERROR(cudaFree(p_separable_column_intermediate_dev));
	HANDLE_ERROR(cudaFree(p_extended_input_dev));

	HANDLE_ERROR(cudaFree(p_kernel_row_dev));
	HANDLE_ERROR(cudaFree(p_kernel_column_dev));

#ifdef _HOST_PIN
	HANDLE_ERROR(cudaFreeHost(p_separable_output_gpu));
#else
	SAFE_FREE(p_separable_output_gpu);
#endif

	SAFE_FREE(p_separable_column_intermediate);
	SAFE_FREE(p_separable_output_cpu);
	SAFE_FREE(p_output);

#ifdef _HOST_PIN
	HANDLE_ERROR(cudaFreeHost(p_extended_input));
#else
	SAFE_FREE(p_extended_input);
#endif

	SAFE_FREE(p_input);
#ifdef _HOST_PIN
	HANDLE_ERROR(cudaFreeHost(p_kernel_row));
	HANDLE_ERROR(cudaFreeHost(p_kernel_column));
#else
	SAFE_FREE(p_kernel_row);
	SAFE_FREE(p_kernel_column);
#endif
	return 0;
}/*main*/