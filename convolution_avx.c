#include "common.h"

#include "convolution_sse4.h"
#include "convolution_serial.h"

#include <immintrin.h>

int ConvolutionAVXMovePtrExtensionCPU(int width, int height,
	float *p_extended_input, int kernel_length, float *p_kernel,
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
		|| NULL == p_kernel
		|| NULL == p_output)
	{
		return -3;
	}

	step_size_avx = sizeof(__m256) / sizeof(float);
	
	steps_avx = kernel_length / step_size_avx;
	remainder_avx = kernel_length % step_size_avx;
	
	steps_size_sse = sizeof(__m128) / sizeof(float);
	steps_sse = remainder_avx / steps_size_sse;
	remainder_sse = remainder_avx % steps_size_sse;

	extended_width = width + kernel_length - 1;

	
	if (kernel_length < step_size_avx)
	{
		return ConvolutionSSE4MovePtrExtensionCPU(
			width, height, p_extended_input, kernel_length, p_kernel, p_output);
	}/*if */


	for (j = 0; j < height; j++) {

		for (i = 0; i < width; i++) {

			int y;
			float *p_mov_kernel;
			float sum;

			int ii, jj;

			sum = 0;
			y = j;

			for (jj = 0; jj < kernel_length; jj++) {

				float *p_mov_input;
				float *p_mov_kernel;

				p_mov_kernel = p_kernel + kernel_length*jj;
				p_mov_input = p_extended_input + y*extended_width + i;

				for (ii = 0; ii < steps_avx; ii++) {

					__m256 m256_kernel, m256_src;
					__m256 m256_temp0;

					float temp_sum;


					m256_kernel = _mm256_loadu_ps(p_mov_kernel);
					m256_src = _mm256_loadu_ps(p_mov_input);

#if(0)
					__m256 m256_temp1, m256_temp2, m256_temp3;
					/*_mm_dp_ps mask :
					upper four bits : whether the corespondent src float be added or not
					lower four bits :  whether the corespondent dest position be stored or not
					*/
					m256_temp1 = _mm256_dp_ps(m256_kernel, m256_src, 0xf1);

					{
						__m128 m128_low, m128_high;
						__m128 m128_sum;

						m128_low = _mm256_castps256_ps128(m256_temp1);
						m128_high = _mm256_extractf128_ps(m256_temp1, 1);
						m128_sum = _mm_add_ps(m128_low, m128_high);
						temp_sum = _mm_cvtss_f32(m128_sum);
					}

#endif

#if(1)
					__m128  m128_temp0, m128_temp1, m128_temp2, m128_temp3,
						m128_temp4, m128_temp5, m128_temp6;

					m256_temp0 = _mm256_mul_ps(m256_kernel, m256_src);
					m128_temp0 = _mm256_castps256_ps128(m256_temp0);
					m128_temp1 = _mm256_extractf128_ps(m256_temp0, 1);

					m128_temp2 = _mm_add_ps(m128_temp0, m128_temp1);

#if(0)				// linker or illegal instruction crash
					m128_temp3 = _mm_movehl_ps(_mm_setzero_ps(), m128_temp2);
#else
					m128_temp3 = _mm_shuffle_ps(m128_temp2, _mm_setzero_ps(),
						_MM_SHUFFLE(0, 0, 3, 2));
#endif
					m128_temp4 = _mm_add_ps(m128_temp2, m128_temp3);

					m128_temp5 = _mm_shuffle_ps(m128_temp4, _mm_setzero_ps(),
						_MM_SHUFFLE(0, 0, 0, 1));

					m128_temp6 = _mm_add_ps(m128_temp4, m128_temp5);
					temp_sum = _mm_cvtss_f32(m128_temp6);
#endif


#if(0)
					__m256 m256_temp1, m256_temp2;

					m256_temp0 = _mm256_mul_ps(m256_kernel, m256_src);
					m256_temp1 = _mm256_hadd_ps(m256_temp0, _mm256_setzero_ps());
					m256_temp2 = _mm256_hadd_ps(m256_temp1, _mm256_setzero_ps());
					{
						__m128 m128_low, m128_high;
						__m128 m128_sum;

						m128_low = _mm256_castps256_ps128(m256_temp2);
						m128_high = _mm256_extractf128_ps(m256_temp2, 1);
						m128_sum = _mm_add_ps(m128_low, m128_high);
						temp_sum = _mm_cvtss_f32(m128_sum);
					}
#endif				

					sum += temp_sum;

					p_mov_kernel += step_size_avx;
					p_mov_input += step_size_avx;
				}/*for ii AVX*/

				for (ii = 0; ii < steps_sse; ii++)
				{
					__m128 m_kernel, m_src;
					__m128 m_temp0, m_temp1;
					float temp_sum;

					m_kernel = _mm_loadu_ps(p_mov_kernel);
					m_src = _mm_loadu_ps(p_mov_input);

					m_temp1 = _mm_dp_ps(m_kernel, m_src, 0xf1);
					temp_sum = _mm_cvtss_f32(m_temp1);

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

}/*ConvolutionAVXMovePtrExtensionCPU*/