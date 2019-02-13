#ifndef _SEPARABLE_CONVOLUTION31_GPU_H_
#define _SEPARABLE_CONVOLUTION31_GPU_H_

#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

	int SeparableConvolutionRowGPUKernelInConstSharedMemPadding31(
		dim3 num_blocks, dim3 num_threads,
		int width, int height, float const *p_extended_input_dev,
		int kernel_length, float const *p_kernel_column_dev,
		float *p_column_done_extended_output_dev);

	int SeparableConvolutionColumnGPUKernelInConstSharedMemPadding31(
		dim3 num_blocks, dim3 num_threads,
		int width, int height, float const *p_column_done_extended_input_dev,
		int kernel_length, float const *p_kernel_row_dev,
		float *p_output_dev);

#ifdef __cplusplus
}
#endif


#endif /*_SEPARABLE_CONVOLUTION31_GPU_H_*/