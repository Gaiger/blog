#ifndef _CONVOLUTION_SERIAL_H_
#define _CONVOLUTION_SERIAL_H_


#ifdef __cplusplus
extern "C" {
#endif


int ConvolutionSerialCPU(int width, int height, float *p_input,
	int kernel_length, float *p_kernel,
	float *p_output);

int ConvolutionSerialExtensionCPU(int width, int height,
	float *p_extended_input, int kernel_length, float *p_kernel,
	float *p_output);

#ifdef __cplusplus
}
#endif

#endif