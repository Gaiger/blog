#ifndef _SEPARABLE_CONVOLUTION_CPU_H_
#define _SEPARABLE_CONVOLUTION_CPU_H_


#ifdef __cplusplus
extern "C" {
#endif

	int SeparableConvolutionColumnSerial(int width, int height, 
		float const *p_extended_input,
		int kernel_length, float const *p_kernel_column,
		float *p_column_done_extended_output);

	int SeparableConvolutionRowSerial(int width, int height,
		float const *p_column_done_extended_input,
		int kernel_length, float const *p_kernel_row,
		float *p_output);


	int SeparableConvolutionColumnSSE4(int width, int height, 
		float const *p_extended_input,
		int kernel_length, float const *p_kernel_column,
		float *p_column_done_extended_output);

	int SeparableConvolutionRowSSE4(int width, int height,
		float const *p_column_done_extended_input,
		int kernel_length, float const *p_kernel_row,
		float *p_output);


	int SeparableConvolutionColumnAVX(int width, int height, 
		float const *p_extended_input,
		int kernel_length, float const *p_kernel_column,
		float *p_column_done_extended_output);
	
	int SeparableConvolutionRowAVX(int width, int height,
		float const *p_column_done_extended_input,
		int kernel_length, float const *p_kernel_row,
		float *p_output);
#ifdef __cplusplus
}
#endif


#endif /*_SEPARABLE_CONVOLUTION_CPU_H_*/