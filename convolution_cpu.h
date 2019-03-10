#ifndef _CONVOLUTION_CPU_H_
#define _CONVOLUTION_CPU_H_

#ifdef __cplusplus
extern "C" {
#endif

int ConvolutionAVX(int width, int height,float const *p_extended_input,
	int kernel_length, float const *p_kernel_matrix, float *p_output);


#ifdef __cplusplus
}
#endif

#endif/*_CONVOLUTION_CPU_H_*/

