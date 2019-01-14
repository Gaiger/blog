
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>



#include "common.h"



#include "convolution_serial.h"
#include "convolution_sse.h"
#include "convolution_sse3.h"
#include "convolution_sse4.h"
#include "convolution_avx.h"

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

	float *p_input_data, *p_extended_data;
	float *p_output_serial;
	float *p_output;
	float *p_kernel_matrix;

	kernel_length = KERNEL_LENGTH;
	kernel_radius = KERNEL_RADIUS;

	width = WIDTH;
	height = HEIGHT;
	extended_width = width + 2 * kernel_radius;
	extended_height = height + 2 * kernel_radius;

	
	p_kernel_matrix = (float*)malloc(
		kernel_length*kernel_length * sizeof(float));
	memset(p_kernel_matrix, 0, 
		kernel_length*kernel_length *sizeof(float));


	p_input_data = (float*)malloc(
		width*height * sizeof(float));

	for (j = 0; j < height; j++) {
		for (i = 0; i < width; i++) {
			p_input_data[j*width + i] = (float)(i + j + 1);
		}/*i*/
	}/*j*/


#if(1)	
	for (j = 0; j < kernel_length; j++) {
		for (i = 0; i < kernel_length; i++) {
			p_kernel_matrix[j*kernel_length + i] = 1;
		}/*i*/
	}/*j*/
#else
	p_kernel_matrix[kernel_length*kernel_radius + kernel_radius] = 1.0;
#endif

	p_extended_data = (float*)malloc(
		extended_width*extended_height * sizeof(float));
	
	p_output = (float*)malloc(
		width*height * sizeof(float));

	p_output_serial = (float*)malloc(
		width*height * sizeof(float));

	/*extend the input data*/
	for (j = 0; j < extended_height; j++) {
		for (i = 0; i < extended_width; i++) {

			if (i < kernel_radius || i >= (kernel_radius + width)
				|| j < kernel_radius || j >= (kernel_radius + height)
				)
			{
				p_extended_data[i + extended_width*j] = 0;
			}
			else
			{
				int ii, jj;
				jj = j - kernel_radius;
				ii = i - kernel_radius;

				p_extended_data[i + extended_width*j]
					= p_input_data[jj*width + ii];
			}/*if */

		}/*for j*/
	}/*for i*/


#if(0)
TIMER_LOOP_BEGIN(CPU_SERIAL, ROUND);
	ConvolutionSerialCPU(width, height, p_input_data,
		kernel_length, p_kernel_matrix, p_output);
TIMER_LOOP_END(CPU_SERIAL);
#endif

TIMER_LOOP_BEGIN(CPU_SERIAL_EXTERNSION, ROUND);
	ConvolutionSerialExtensionCPU(width, height, p_extended_data,
		kernel_length, p_kernel_matrix, p_output);
TIMER_LOOP_END(CPU_SERIAL_EXTERNSION);

	memcpy(p_output_serial, p_output, 
		width*height * sizeof(float));

#if(0)
TIMER_LOOP_BEGIN(CPU_SSE_EXTERNSION, ROUND);
	ConvolutionSSEExtensionCPU(width, height, p_extended_data,
		kernel_length, p_kernel_matrix, p_output);
TIMER_LOOP_END(CPU_SSE_EXTERNSION);
#endif

#if(1)
TIMER_LOOP_BEGIN(CPU_SSE_MOVPTR_EXTERNSION, ROUND);
	ConvolutionSSEMovePtrExtensionCPU(width, height, p_extended_data,
		kernel_length, p_kernel_matrix, p_output);
TIMER_LOOP_END(CPU_SSE_MOVPTR_EXTERNSION);
#endif


#if(1)
TIMER_LOOP_BEGIN(CPU_SSE3_MOVPTR_EXTERNSION, ROUND);
	ConvolutionSSE3MovePtrExtensionCPU(width, height, p_extended_data,
	kernel_length, p_kernel_matrix, p_output);
TIMER_LOOP_END(CPU_SSE3_MOVPTR_EXTERNSION);
#endif


#if(1)
TIMER_LOOP_BEGIN(CPU_SSE4_MOVPTR_EXTERNSION, ROUND);
	ConvolutionSSE4MovePtrExtensionCPU(width, height, p_extended_data,
		kernel_length, p_kernel_matrix, p_output);
TIMER_LOOP_END(CPU_SSE4_MOVPTR_EXTERNSION);
#endif


#if(1)
TIMER_LOOP_BEGIN(CPU_AVX_MOVPTR_EXTERNSION, ROUND);
	ConvolutionAVXMovePtrExtensionCPU(width, height, p_extended_data,
		kernel_length, p_kernel_matrix, p_output);
TIMER_LOOP_END(CPU_AVX_MOVPTR_EXTERNSION);
#endif

#if(1)
	int count = 0;
	for (j = 0; j < height; j++) {
		for (i = 0; i < width; i++) {

			int is_over_tolerance;
			is_over_tolerance = 0;
			if (0 == p_output_serial[j*width + i])
			{
				if (0 != p_output[j*width + i])
					is_over_tolerance = 1;
			}
			else
			{
				float tolerance;
				tolerance = p_output_serial[j*width + i] - p_output[j*width + i];
				tolerance /= p_output_serial[j*width + i];
				tolerance = (float)fabs(tolerance);
				if (tolerance > 1.0e-6)
					is_over_tolerance = 1;
			}

			if (0 != is_over_tolerance)
			{
				printf("computs error : i = %d, j = %d"
					", value = %.f %.f\r\n", i, j, p_output_serial[j*width + i]
					, p_output[j*width + i]);
				count++;
			}

		}/*for i*/
	}/*for j*/

	if (0 <count)
		printf("error count = %d\r\n", count);
#endif

	SAFE_FREE(p_output);
	SAFE_FREE(p_output_serial);
	SAFE_FREE(p_kernel_matrix);
	SAFE_FREE(p_input_data);
	SAFE_FREE(p_extended_data);

	return 0;
}/*main*/