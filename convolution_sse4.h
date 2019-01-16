#ifndef _CONVOLUTION_SSE4_H_
#define _CONVOLUTION_SSE4_H_

#ifdef __cplusplus
extern "C" {
#endif


int ConvolutionSSE4MovePtrExtensionCPU(int width, int height,
	float *p_extended_input, int kernel_length, float *p_kernel,
	float *p_output);

#if(21 == KERNEL_LENGTH)
int ConvolutionSSE4MovePtrUnrollKernelLengh21ExtensionCPU(int width, int height,
	float *p_extended_input, int kernel_length, float *p_kernel,
	float *p_output);


int ConvolutionSSE4MovePtrUnrollKernelLengh21AlignmentExtensionCPU(
	int width, int height,
	float *p_extended_input, int kernel_length, float *p_kernel,
	float *p_output);

#endif/*21 == KERNEL_LENGTH*/
#ifdef __cplusplus
}
#endif

#endif /*_CONVOLUTION_SSE4_H_*/
