#ifndef _CONVOLUTION_AVX_H_
#define _CONVOLUTION_AVX_H_

#ifdef __cplusplus
extern "C" {
#endif


int ConvolutionAVXDotMovePtrExtensionCPU(int width, int height,
	float *p_extended_input, int kernel_length, float *p_kernel,
	float *p_output);

int ConvolutionAVXShuMovePtrExtensionCPU(int width, int height,
	float *p_extended_input, int kernel_length, float *p_kernel,
	float *p_output);

int ConvolutionAVXHAddMovePtrExtensionCPU(int width, int height,
	float *p_extended_input, int kernel_length, float *p_kernel,
	float *p_output);

#if(21 == KERNEL_LENGTH)
int ConvolutionAVXHAddMovePtrUnrollKernelLengh21AlignmentExtensionCPU(int width, int height,
	float *p_extended_input, int kernel_length, float *p_kernel,
	float *p_output);
#endif
#ifdef __cplusplus
}
#endif

#endif /*_CONVOLUTION_AVX_H_*/