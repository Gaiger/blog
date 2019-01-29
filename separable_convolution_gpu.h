#ifndef _SEPARABLE_CONVOLUTION_GPU_H_
#define _SEPARABLE_CONVOLUTION_GPU_H_

#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

	int SeparableConvolutionRowGPULinearMemory(
		dim3 num_blocks, dim3 num_threads,
		int width, int height, float const *p_extended_input_dev,
		int kernel_length, float const *p_kernel_row_dev,
		float *p_row_done_extended_output_dev);

	int SeparableConvolutionColumnGPULinearMemory(
		dim3 num_blocks, dim3 num_threads,
		int width, int height, float const *p_row_done_extended_input_dev,
		int kernel_length, float const *p_kernel_column_dev,
		float *p_output_dev);


	int SeparableConvolutionRowGPUKernelInConst(
		dim3 num_blocks, dim3 num_threads,
		int width, int height, float const *p_extended_input_dev,
		int kernel_length, float const *p_kernel_row_host,
		float *p_row_done_extended_output_dev);

	int SeparableConvolutionColumnGPUKernelInConst(
		dim3 num_blocks, dim3 num_threads,
		int width, int height, float const *p_row_done_extended_input_dev,
		int kernel_length, float const *p_kernel_column_host,
		float *p_output_dev);


	int SeparableConvolutionRowGPUKernelInConstSharedMem(
		dim3 num_blocks, dim3 num_threads,
		int width, int height, float const *p_extended_input_dev,
		int kernel_length, float const *p_kernel_row_host,
		float *p_row_done_extended_output_dev);

	int SeparableConvolutionColumnGPUKernelInConstSharedMem(
		dim3 num_blocks, dim3 num_threads,
		int width, int height, float const *p_row_done_extended_input_dev,
		int kernel_length, float const *p_kernel_column_host,
		float *p_output_dev);


	int SeparableConvolutionRowGPUKernelInConstSharedMemPadding(
		dim3 num_blocks, dim3 num_threads,
		int width, int height, float const *p_extended_input_dev,
		int kernel_length, float const *p_kernel_row_host,
		float *p_row_done_extended_output_dev);

	int SeparableConvolutionColumnGPUKernelInConstSharedMemPadding(
		dim3 num_blocks, dim3 num_threads,
		int width, int height, float const *p_row_done_extended_input_dev,
		int kernel_length, float const *p_kernel_column_host,
		float *p_output_dev);

#ifdef __cplusplus
}
#endif


#endif /*_SEPARABLE_CONVOLUTION_GPU_H_*/