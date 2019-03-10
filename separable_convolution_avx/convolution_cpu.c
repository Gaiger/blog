
#include "convolution_cpu.h"


#pragma warning(disable:4752)
#include <immintrin.h>

#define LOCAL						static

LOCAL int ConvolutionSerial(int width, int height,
	float const *p_extended_input, int kernel_length, float const *p_kernel_matrix,
	float *p_output)
{
	int i, j;
	int extended_width;

	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */

	if (NULL == p_extended_input
		|| NULL == p_kernel_matrix
		|| NULL == p_output)
	{
		return -3;
	}

	extended_width = width + kernel_length - 1;

	for (j = 0; j < height; j++) {
		for (i = 0; i < width; i++) {

			int ii, jj;
			float sum;
			sum = 0;

			for (jj = 0; jj < kernel_length; jj++) {

				int y_mul_kernel_width, y_mul_input_width;

				y_mul_kernel_width = jj*kernel_length;
				y_mul_input_width = (j + jj)*extended_width;

				for (ii = 0; ii < kernel_length; ii++) {
					int x;
					x = i + ii;
				
					sum += p_kernel_matrix[y_mul_kernel_width + ii]
						* p_extended_input[y_mul_input_width + x];
				}/*for ii*/
			}/*for jj*/

			p_output[j*width + i] = sum;
		}/*for i*/
	}/*for j*/

	return 0;
}/*ConvolutionSerial*/


LOCAL int ConvolutionSSE4CPU(int width, int height,
	float const *p_extended_input, int kernel_length, float const *p_kernel_matrix,
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
		|| NULL == p_kernel_matrix
		|| NULL == p_output)
	{
		return -3;
	}

	step_size = sizeof(__m128) / sizeof(float);

	extended_width = width + kernel_length - 1;
	steps = kernel_length / step_size;

	if (kernel_length < step_size)
	{
		return ConvolutionSerial(width, height, p_extended_input,
			kernel_length, p_kernel_matrix, p_output);
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

				p_mov_kernel = (float*)p_kernel_matrix + kernel_length*jj;
				p_mov_input = (float*)p_extended_input + y*extended_width + i;

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

}/*ConvolutionSSE4CPU*/


int ConvolutionAVX(int width, int height,
	float const *p_extended_input, int kernel_length, float const *p_kernel_matrix,
	float *p_output)
{
	int i, j;

	int extended_width;

	int step_size_avx;
	int steps_avx;
	int remainder_avx;

	int steps_size_sse;
	int steps_sse;
	int remainder_sse;


	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */

	if (NULL == p_extended_input
		|| NULL == p_kernel_matrix
		|| NULL == p_output)
	{
		return -3;
	}

	step_size_avx = sizeof(__m256) / sizeof(float);

	if (kernel_length < step_size_avx)
	{
		return ConvolutionSSE4CPU(
			width, height, p_extended_input, 
			kernel_length, p_kernel_matrix, p_output);
	}/*if */

	steps_avx = kernel_length / step_size_avx;
	remainder_avx = kernel_length % step_size_avx;

	steps_size_sse = sizeof(__m128) / sizeof(float);
	steps_sse = remainder_avx / steps_size_sse;
	remainder_sse = remainder_avx % steps_size_sse;

	extended_width = width + kernel_length - 1;


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

				p_mov_kernel = (float*)p_kernel_matrix + kernel_length*jj;
				p_mov_input = (float*)p_extended_input + y*extended_width + i;

				for (ii = 0; ii < steps_avx; ii++) {

					__m256 m256_kernel, m256_src;

					float temp_sum;


					m256_kernel = _mm256_loadu_ps(p_mov_kernel);
					m256_src = _mm256_loadu_ps(p_mov_input);

					{
						__m256 m256_temp0;
						/*_mm_dp_ps mask :
						upper four bits : whether the corespondent src float be added or not
						lower four bits :  whether the corespondent dest position be stored or not
						*/
						m256_temp0 = _mm256_dp_ps(m256_kernel, m256_src, 0xf1);

						{
							__m128 m128_low, m128_high;
							__m128 m128_sum;

							m128_low = _mm256_castps256_ps128(m256_temp0);
							m128_high = _mm256_extractf128_ps(m256_temp0, 1);
							m128_sum = _mm_add_ps(m128_low, m128_high);
							temp_sum = _mm_cvtss_f32(m128_sum);
						}
					}/*local variable*/

					sum += temp_sum;

					p_mov_kernel += step_size_avx;
					p_mov_input += step_size_avx;
				}/*for ii AVX*/

				for (ii = 0; ii < steps_sse; ii++)
				{
					__m128 m128_kernel, m128_src;
					__m128 m128_temp0;
					float temp_sum;

					m128_kernel = _mm_loadu_ps(p_mov_kernel);
					m128_src = _mm_loadu_ps(p_mov_input);

					m128_temp0 = _mm_dp_ps(m128_kernel, m128_src, 0xf1);
					temp_sum = _mm_cvtss_f32(m128_temp0);

					sum += temp_sum;
					p_mov_kernel += steps_size_sse;
					p_mov_input += steps_size_sse;
				}/*(kernel_length%8) /4 > 0*/

				{
					int serial_begin;
					serial_begin = steps_avx*step_size_avx
						+ steps_size_sse * steps_sse;

					for (ii = serial_begin; ii < kernel_length; ii++) {
						sum += p_mov_kernel[0] * p_mov_input[0];
						p_mov_kernel += 1;
						p_mov_input += 1;
					}/*for ii*/
				}/* kernel_length%4 */

				y += 1;
			}/*for jj*/

			p_output[j*width + i] = sum;
		}/*for i*/
	}/*for j*/

	return 0;

}/*ConvolutionAVX*/
