#ifndef _CONVOLUTION_SSE3_H_
#define _CONVOLUTION_SSE3_H_

#ifdef __cplusplus
extern "C" {
#endif



int ConvolutionSSE3MovePtrExtensionCPU(int width, int height,
	float *p_extended_input, int kernel_length, float *p_kernel,
	float *p_output);


#ifdef __cplusplus
}
#endif

#endif /*_CONVOLUTION_SSE3_H_*/
