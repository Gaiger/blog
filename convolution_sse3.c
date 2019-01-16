#include "common.h"
#include "convolution_sse3.h"

#include "convolution_serial.h"

#include <pmmintrin.h>



int ConvolutionSSE3HAddMovePtrExtensionCPU(int width, int height,
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

			int ii, jj;			
			float sum;
			int y;

			sum = 0;

			y = j;
			for (jj = 0; jj < kernel_length; jj++) {


				float *p_mov_input;
				float *p_mov_kernel;

				p_mov_kernel = p_kernel + kernel_length*jj;
				p_mov_input = p_extended_input + y*extended_width + i;
				for (ii = 0; ii < steps; ii++) {

					__m128 m_kernel, m_src;
					__m128 m_temp0, m_temp1, m_temp2;
					float temp_sum;
					
					m_kernel = _mm_loadu_ps(p_mov_kernel);

					m_src = _mm_loadu_ps(p_mov_input);
					
					m_temp0 = _mm_mul_ps(m_kernel, m_src);

					m_temp1  =  _mm_hadd_ps(m_temp0, _mm_setzero_ps());
					m_temp2 = _mm_hadd_ps(m_temp1, _mm_setzero_ps());
					temp_sum = _mm_cvtss_f32(m_temp2);


					sum += temp_sum;

					p_mov_kernel += step_size;
					p_mov_input += step_size;
				}/*for ii*/

				for (ii = steps*step_size; ii < kernel_length; ii++) {
					sum += p_mov_kernel[0] * p_mov_input[0];
					p_mov_kernel += 1;
					p_mov_input += 1;
				}/*for ii kernel_length*/

				y += 1;
			}/*for jj*/

			p_output[j*width + i] = sum;
		}/*for i*/
	}/*for j*/

	return 0;

}/*ConvolutionSSE3MovePtrExtensionCPU*/



int ConvolutionSSE3ShuMovePtrExtensionCPU(int width, int height,
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

			int ii, jj;
			float sum;
			int y;

			sum = 0;

			y = j;
			for (jj = 0; jj < kernel_length; jj++) {


				float *p_mov_input;
				float *p_mov_kernel;

				p_mov_kernel = p_kernel + kernel_length*jj;
				p_mov_input = p_extended_input + y*extended_width + i;
				for (ii = 0; ii < steps; ii++) {

					__m128 m_kernel, m_src, m_dst;
					__m128 m_temp0, m_temp1, m_temp2, m_temp3;
					float temp_sum;

					m_kernel = _mm_loadu_ps(p_mov_kernel);

					m_src = _mm_loadu_ps(p_mov_input);

					m_temp0 = _mm_mul_ps(m_kernel, m_src);


					/*
					temp0 = a0, a1, a2, a3
					*/
					m_temp1 = _mm_movehdup_ps(m_temp0); //m_temp1 = a1, a1, a3, a3
					m_temp2 = _mm_add_ps(m_temp0, m_temp1); //m_temp2 = a1+a0, a1+ a1, a2 + a3, a3 + a3,
					m_temp3 = _mm_movehl_ps(_mm_setzero_ps(), m_temp2); //m_temp3 = a2 + a3, a3 + a3, 0, 0
					m_dst = _mm_add_ps(m_temp1, m_temp3);
					temp_sum = _mm_cvtss_f32(m_dst);

					sum += temp_sum;

					p_mov_kernel += step_size;
					p_mov_input += step_size;
				}/*for ii*/

				for (ii = steps*step_size; ii < kernel_length; ii++) {
					sum += p_mov_kernel[0] * p_mov_input[0];
					p_mov_kernel += 1;
					p_mov_input += 1;
				}/*for ii kernel_length*/

				y += 1;
			}/*for jj*/

			p_output[j*width + i] = sum;
		}/*for i*/
	}/*for j*/

	return 0;

}/*ConvolutionSSE3MovePtrExtensionCPU*/


