#include "common.h"
#include "convolution_sse3.h"

#include "convolution_serial.h"

#include <smmintrin.h>



int ConvolutionSSE4MovePtrExtensionCPU(int width, int height,
	float *p_extended_input, int kernel_length, float *p_kernel,
	float *p_output)
{
	int i, j;

	int extended_width;

	int step_size;
	int steps;


	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */

	if (NULL == p_extended_input
		|| NULL == p_kernel
		|| NULL == p_output)
	{
		return -3;
	}

	step_size = sizeof(__m128) / sizeof(float);

	extended_width = width + kernel_length - 1;
	steps = kernel_length / step_size;

	if (kernel_length < step_size)
	{
		return ConvolutionSerialExtensionCPU(width, height, p_extended_input,
			kernel_length, p_kernel, p_output);
	}/*if */


	for (j = 0; j < height; j++) {

		for (i = 0; i < width; i++) {

			int y;
			float sum;
			
			int ii, jj;

			sum = 0;
			y = j;

			for (jj = 0; jj < kernel_length; jj++) {

				float *p_mov_input;
				float *p_mov_kernel;

				p_mov_kernel = p_kernel + kernel_length*jj;
				p_mov_input = p_extended_input + y*extended_width + i;

				for (ii = 0; ii < steps; ii++) {

					__m128 m_kernel, m_src;
					__m128 m_temp0;
					float temp_sum;
					
					m_kernel = _mm_loadu_ps(p_mov_kernel);
					m_src = _mm_loadu_ps(p_mov_input);

					/*_mm_dp_ps mask :
						upper four bits : whether the corespondent src float be added or not
						lower four bits :  whether the corespondent dest position be stored or not
					*/
					m_temp0 = _mm_dp_ps(m_kernel, m_src, 0xf1);
					temp_sum = _mm_cvtss_f32(m_temp0);


					sum += temp_sum;

					p_mov_kernel += step_size;
					p_mov_input += step_size;
				}/*for ii*/


				for (ii = steps*step_size; ii < kernel_length; ii++) {
					sum += p_mov_kernel[0] * p_mov_input[0];
					p_mov_kernel += 1;
					p_mov_input += 1;
				}/*for ii*/

				y += 1;
			}/*for jj*/

			p_output[j*width + i] = sum;
		}/*for i*/
	}/*for j*/

	return 0;

}/*ConvolutionSSE4MovePtrExtensionCPU*/

#if(21 == KERNEL_LENGTH)

int ConvolutionSSE4MovePtrUnrollKernelLengh21ExtensionCPU(int width, int height,
	float *p_extended_input, int kernel_length, float *p_kernel,
	float *p_output)
{
	int i, j;

	int extended_width;

	int step_size;
	int steps;


	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */

	if (NULL == p_extended_input
		|| NULL == p_kernel
		|| NULL == p_output)
	{
		return -3;
	}

	step_size = sizeof(__m128) / sizeof(float);

	extended_width = width + kernel_length - 1;
	steps = kernel_length / step_size;

	if (kernel_length < step_size)
	{
		return ConvolutionSerialExtensionCPU(width, height, p_extended_input,
			kernel_length, p_kernel, p_output);
	}/*if */


	for (j = 0; j < height; j++) {

		for (i = 0; i < width; i++) {

			int y;
			float sum;

			int jj;

			sum = 0;
			y = j;

			for (jj = 0; jj < kernel_length; jj++) {

				float *p_mov_input;
				float *p_mov_kernel;

				__m128 m_kernel, m_src;
				__m128 m_temp0;

				p_mov_kernel = p_kernel + kernel_length*jj;
				p_mov_input = p_extended_input + y*extended_width + i;
				
#define ONE_SSE4_STEP()					\
				{						\
					float temp_sum;			\
											\
					m_kernel = _mm_loadu_ps(p_mov_kernel); \
					m_src = _mm_loadu_ps(p_mov_input); \
					\
					\
					m_temp0 = _mm_dp_ps(m_kernel, m_src, 0xf1); \
					temp_sum = _mm_cvtss_f32(m_temp0); \
					\
					sum += temp_sum; \
										\
					p_mov_kernel += sizeof(__m128) / sizeof(float); \
					p_mov_input += sizeof(__m128) / sizeof(float); \
				}	

				ONE_SSE4_STEP();
				ONE_SSE4_STEP();
				ONE_SSE4_STEP();
				ONE_SSE4_STEP();
				ONE_SSE4_STEP();
				{
					sum += p_mov_kernel[0] * p_mov_input[0];
				}/*for ii*/

				y += 1;
			}/*for jj*/

			p_output[j*width + i] = sum;
		}/*for i*/
	}/*for j*/

	return 0;

}/*ConvolutionSSE4MovePtrUnrollKernelLengh21ExtensionCPU*/



int ConvolutionSSE4MovePtrUnrollKernelLengh21AlignmentExtensionCPU(
	int width, int height,
	float *p_extended_input, int kernel_length, float *p_kernel,
	float *p_output)
{
	int i, j;

	int extended_width;

	int step_size;
	int steps;
	int kernel_length_alignment16;

	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */

	if (NULL == p_extended_input
		|| NULL == p_kernel
		|| NULL == p_output)
	{
		return -3;
	}

	step_size = sizeof(__m128) / sizeof(float);

	extended_width = width + kernel_length - 1;
	steps = kernel_length / step_size;

	if (kernel_length < step_size)
	{
		return ConvolutionSerialExtensionCPU(width, height, p_extended_input,
			kernel_length, p_kernel, p_output);
	}/*if */

	kernel_length_alignment16 
		= (1 + kernel_length * sizeof(float) / 16)*(16/sizeof(float));

	for (j = 0; j < height; j++) {

		for (i = 0; i < width; i++) {

			int y;
			float sum;

			int jj;

			sum = 0;
			y = j;

			for (jj = 0; jj < kernel_length; jj++) {

				float *p_mov_input;
				float *p_mov_kernel;

				__m128 m_kernel, m_src;
				__m128 m_temp0;

				p_mov_kernel = p_kernel + kernel_length_alignment16*jj;
				p_mov_input = p_extended_input + y*extended_width + i;

#define ONE_SSE4_KERNEL_ALIGNED16_STEP()			\
				{							\
					float temp_sum;			\
											\
					m_kernel = _mm_load_ps(p_mov_kernel); \
					m_src = _mm_loadu_ps(p_mov_input); \
					\
					\
					m_temp0 = _mm_dp_ps(m_kernel, m_src, 0xf1); \
					temp_sum = _mm_cvtss_f32(m_temp0); \
					\
					sum += temp_sum; \
					\
					p_mov_kernel += sizeof(__m128) / sizeof(float); \
					p_mov_input += sizeof(__m128) / sizeof(float); \
				}

				ONE_SSE4_KERNEL_ALIGNED16_STEP();
				ONE_SSE4_KERNEL_ALIGNED16_STEP();
				ONE_SSE4_KERNEL_ALIGNED16_STEP();
				ONE_SSE4_KERNEL_ALIGNED16_STEP();
				ONE_SSE4_KERNEL_ALIGNED16_STEP();

				{
					sum += p_mov_kernel[0] * p_mov_input[0];
				}/*for ii*/

				y += 1;
			}/*for jj*/

			p_output[j*width + i] = sum;
		}/*for i*/
	}/*for j*/

	return 0;

}/*ConvolutionSSE4MovePtrUnrollKernelLengh21AlignmentExtensionCPU*/
#endif