#ifndef _SEPARABLE_CONVOLUTION_CPU_H_
#define _SEPARABLE_CONVOLUTION_CPU_H_


#ifdef __cplusplus
extern "C" {
#endif

	int SeparateConvolutionRowSerial(int width, int height, float const *p_extended_input,
		int kernel_length, float const *p_kernel_row,
		float *p_row_done_extended_output);

	int SeparateConvolutionColumnSerial(int width, int height,
		float const *p_row_done_extended_input,
		int kernel_length, float const *p_kernel_column,
		float *p_output);


	int SeparateConvolutionRowSSE4(int width, int height, float const *p_extended_input,
		int kernel_length, float const *p_kernel_row,
		float *p_row_done_extended_output);

	int SeparateConvolutionColumnSSE4(int width, int height,
		float const *p_row_done_extended_input,
		int kernel_length, float const *p_kernel_column,
		float *p_output);


	int SeparateConvolutionRowAVX(int width, int height, float const *p_extended_input,
		int kernel_length, float const *p_kernel_row,
		float *p_row_done_extended_output);
	
	int SeparateConvolutionColumnAVX(int width, int height,
		float const *p_row_done_extended_input,
		int kernel_length, float const *p_kernel_column,
		float *p_output);
#ifdef __cplusplus
}
#endif


#endif /*_SEPARABLE_CONVOLUTION_CPU_H_*/