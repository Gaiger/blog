
#include <stdio.h>
#include <math.h>

#include "common.h"

#include "convolution_cpu.h"
#include "separable_convolution_cpu.h"


#ifdef _MSC_VER 

#include <windows.h>
//#include <mmsystem.h>


unsigned int GetTime(void)
{
	return (unsigned int)timeGetTime();
}/*GetTime*/

#endif


#define SAFE_FREE(PTR)				if(NULL != (PTR)) \
										free(PTR);	\
									(PTR) = NULL;


#define TIMER_BEGIN(TIMER_LABEL)	\
									{	unsigned int tBegin##TIMER_LABEL, tEnd##TIMER_LABEL; \
										tBegin##TIMER_LABEL = GetTime();

#define TIMER_END(TIMER_LABEL)		\
										tEnd##TIMER_LABEL = GetTime();\
										fprintf(stderr, "%s cost time = %d ms\n\n", \
											#TIMER_LABEL, tEnd##TIMER_LABEL - tBegin##TIMER_LABEL);	\
									}

#define TIMER_LOOP_BEGIN(TIMER_LABEL, N_ROUND)	 \
									fprintf(stderr, "ROUND = %d\n", N_ROUND); \
									TIMER_BEGIN(TIMER_LABEL);	\
									for(int ijk = 0; ijk <(N_ROUND); ijk++){


#define TIMER_LOOP_END(TIMER_LABEL)	 \
									}		\
									TIMER_END(TIMER_LABEL); 


int main(int argc, char *argv[])
{
	int i, j;
	int width, height;
	int extended_width, extended_height;

	int kernel_length, kernel_radius;

	float *p_kernel_row;
	float *p_kernel_column;
	float *p_kernel_matrix;

	float *p_input, *p_extended_input;
	float *p_output;
	float *p_separate_row_intermediate;
	float *p_separate_output_cpu;


	printf("input = %d x %d\r\n", WIDTH, HEIGHT);
	printf("kernel :: %d x %d\r\n",
		(2* KERNEL_RADIUS + 1), (2 * KERNEL_RADIUS + 1));

	width = WIDTH;
	height = HEIGHT;
	kernel_radius = KERNEL_RADIUS;
	extended_width = width + 2 * kernel_radius;
	extended_height = height + 2 * kernel_radius;
	kernel_length = KERNEL_LENGTH;

#ifdef _KERNEL_ALIGNED16
	p_kernel_row = (float*)_aligned_malloc(kernel_length * sizeof(float), 
		16);
	p_kernel_column = (float*)_aligned_malloc(kernel_length * sizeof(float), 
		16);
#else
	p_kernel_row = (float*)malloc(kernel_length * sizeof(float));
	p_kernel_column = (float*)malloc(kernel_length * sizeof(float));
#endif
	{
		float shift_value;
		for (i = 0; i < kernel_length; i++)
			p_kernel_row[i] = (float)i + 1;

		shift_value = p_kernel_row[kernel_length - 1];

		for (i = 0; i < kernel_length; i++)
			p_kernel_column[i] = (float)i + shift_value + 1;
	
	}/*local variable*/


	p_kernel_matrix = (float*)malloc(kernel_length*kernel_length * sizeof(float));
	
	for (j = 0; j < kernel_length; j++) {
		for (i = 0; i < kernel_length; i++) {
			p_kernel_matrix[j*kernel_length + i]
				= p_kernel_row[j] * p_kernel_column[i];
		}/*i*/
	}/*j*/

	p_input = (float*)malloc(width*height * sizeof(float));
	for (j = 0; j < height; j++) {
		for (i = 0; i < width; i++) {
			p_input[j*width + i] = (float)(i + j*width + 1);
		}/*i*/
	}/*j*/

	p_extended_input = (float*)malloc(
		extended_width*extended_height * sizeof(float));

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
#if(1)
	TIMER_LOOP_BEGIN(CONVOLUTION_AVX, ROUND)
		ConvolutionAVX(width, height, p_extended_input,
			kernel_length, p_kernel_matrix, p_output);
	TIMER_LOOP_END(CONVOLUTION_AVX)
#endif
	p_separate_row_intermediate = (float*)malloc(
		extended_width*height * sizeof(float));

	p_separate_output_cpu = (float*)malloc(
		width*height * sizeof(float));
#if(1)
TIMER_LOOP_BEGIN(SEPAREATE_CONVOLUTION_SERIAL, ROUND)
		SeparateConvolutionRowSerial(width, height, p_extended_input,
			kernel_length, p_kernel_row, p_separate_row_intermediate);

	SeparateConvolutionColumnSerial(width, height, p_separate_row_intermediate,
		kernel_length, p_kernel_column, p_separate_output_cpu);
TIMER_LOOP_END(SEPAREATE_CONVOLUTION_SERIAL)
#endif

#if(1)
TIMER_LOOP_BEGIN(SEPAREATE_CONVOLUTION_SSE4, ROUND)
	SeparateConvolutionRowSSE4(width, height, p_extended_input,
		kernel_length, p_kernel_row, p_separate_row_intermediate);

	SeparateConvolutionColumnSSE4(width, height, p_separate_row_intermediate,
		kernel_length, p_kernel_column, p_separate_output_cpu);
TIMER_LOOP_END(SEPAREATE_CONVOLUTION_SSE4)
#endif

#if(1)
TIMER_LOOP_BEGIN(SEPAREATE_CONVOLUTION_AVX, ROUND)
	SeparateConvolutionRowAVX(width, height, p_extended_input,
		kernel_length, p_kernel_row, p_separate_row_intermediate);

	SeparateConvolutionColumnAVX(width, height, p_separate_row_intermediate,
		kernel_length, p_kernel_column, p_separate_output_cpu);
TIMER_LOOP_END(SEPAREATE_CONVOLUTION_AVX)
#endif


#if(1)
	int count = 0;
	for (j = 0; j < height; j++) {
		for (i = 0; i < width; i++) {

			int is_over_tolerance;
			is_over_tolerance = 0;
			if (0 == p_output[j*width + i])
			{
				if (0 != p_separate_output_cpu[j*width + i])
					is_over_tolerance = 1;
			}
			else
			{
				float tolerance;
				tolerance = p_separate_output_cpu[j*width + i] - p_output[j*width + i];
				tolerance /= p_output[j*width + i];
				tolerance = (float)fabs(tolerance);
				if (tolerance > 1.0e-6)
					is_over_tolerance = 1;
			}

			if (0 != is_over_tolerance)
			{
				printf("computs error : i = %d, j = %d"
					", value = %.f %.f\r\n", i, j, p_separate_output_cpu[j*width + i]
					, p_output[j*width + i]);
				count++;
			}

		}/*for i*/
	}/*for j*/

	if (0 <count)
		printf("error count = %d\r\n", count);
#endif
	


	SAFE_FREE(p_separate_row_intermediate);
	SAFE_FREE(p_separate_output_cpu);
	SAFE_FREE(p_output);
	
	SAFE_FREE(p_extended_input);
	SAFE_FREE(p_input);

	SAFE_FREE(p_kernel_matrix);

#ifdef _KERNEL_ALIGNED16
	_aligned_free(p_kernel_row);
	_aligned_free(p_kernel_column);
#else
	SAFE_FREE(p_kernel_row);
	SAFE_FREE(p_kernel_column);
#endif	
	return 0;
}/*main*/