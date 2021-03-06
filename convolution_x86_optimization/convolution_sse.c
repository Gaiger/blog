#include "common.h"
#include "convolution_sse.h"

#include "convolution_serial.h"

#include <xmmintrin.h>


int ConvolutionSSEExtensionCPU(int width, int height,
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
			sum = 0;
			

			for (jj = 0; jj < kernel_length; jj++) {
				for (ii = 0; ii < steps; ii++) {
					
					__m128 m_kernel, m_src;
					__m128 m_temp0, m_temp1, m_temp2, m_temp3, m_temp4;
					float temp_sum;

					int x, y;
					x = i + ii*step_size;
					y = j + jj;

					m_kernel = _mm_loadu_ps(p_kernel + kernel_length*jj + ii*step_size);

					m_src = _mm_loadu_ps(p_extended_input + y*extended_width + x);

					m_temp0 = _mm_mul_ps(m_kernel, m_src);

					/*
					_mm_shuffle_ps (a0..a3, b0..b3, _MM_SHUFFLE(bx, bx, ax, ax));
					ex _MM_SHUFFLE(3, 2, 0, 1) out : a1 a0 b2 b3
					*/
#if(1)
					m_temp1 = _mm_movehl_ps(_mm_setzero_ps(), m_temp0);
#else
					m_temp1 = _mm_shuffle_ps(m_temp0, _mm_setzero_ps(),
						_MM_SHUFFLE(0, 0, 3, 2));
#endif
					m_temp2 = _mm_add_ps(m_temp0, m_temp1);
					m_temp3 = _mm_shuffle_ps(m_temp2, _mm_setzero_ps(),
						_MM_SHUFFLE(0, 0, 0, 1));

					m_temp4 = _mm_add_ps(m_temp3, m_temp2);
#if(1)
					temp_sum = _mm_cvtss_f32(m_temp4);
#else
					_mm_store_ss(&temp_sum, m_temp4);
#endif
					sum += temp_sum;
				}/*for ii*/

				for (ii = steps*step_size; ii < kernel_length; ii++) {
					int x, y;

					x = i + ii;
					y = j + jj;

					sum += p_kernel[kernel_length*jj + ii]
						* p_extended_input[extended_width*y + x];
				}
			}/*for jj*/

			p_output[j*width + i] = sum;
		}/*for i*/
	}/*for j*/

	return 0;

}/*ConvolutionSSEExternionCPU*/


int ConvolutionSSEMovePtrExtensionCPU(int width, int height,
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
					__m128 m_temp0, m_temp1, m_temp2, m_temp3, m_temp4;
					float temp_sum;

					m_kernel = _mm_loadu_ps(p_mov_kernel);
					m_src = _mm_loadu_ps(p_mov_input);

					m_temp0 = _mm_mul_ps(m_kernel, m_src);
#if(1)
					m_temp1 = _mm_movehl_ps(_mm_setzero_ps(), m_temp0);
#else
					m_temp1 = _mm_shuffle_ps(m_temp0, _mm_setzero_ps(),
						_MM_SHUFFLE(0, 0, 3, 2));
#endif
					m_temp2 = _mm_add_ps(m_temp0, m_temp1);
					m_temp3 = _mm_shuffle_ps(m_temp2, _mm_setzero_ps(),
						_MM_SHUFFLE(0, 0, 0, 1));

					m_temp4 = _mm_add_ps(m_temp3, m_temp2);

#if(1)
					temp_sum = _mm_cvtss_f32(m_temp4);
#else
					_mm_store_ss(&temp_sum, m_temp4);
#endif
					sum += temp_sum;

					p_mov_kernel += step_size;
					p_mov_input += step_size;
				}/*for ii*/

				for (ii = steps*step_size; ii < kernel_length; ii++) {

					sum += p_mov_kernel[0]*p_mov_input[0];
					p_mov_kernel += 1;
					p_mov_input += 1;
				}/*for ii kernel_length*/

				y += 1;
			}/*for jj*/

			p_output[j*width + i] = sum;
		}/*for i*/
	}/*for j*/

	return 0;
}/*ConvolutionSSEMovePtrExternionCPU*/