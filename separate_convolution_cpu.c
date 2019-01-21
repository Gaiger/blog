#include <stdlib.h>
#include <string.h>

#pragma warning(disable:4752)
#include <immintrin.h>

#include "common.h"


#ifdef _KERNEL_ALIGNED16
#define __MM128_LOAD_KERNEL(ADDR)			_mm_load_ps((ADDR))
#define __MM256_LOAD_KERNEL(ADDR)			_mm256_load_ps((ADDR))
#else
#define __MM128_LOAD_KERNEL(ADDR)			_mm_loadu_ps((ADDR))
#define __MM256_LOAD_KERNEL(ADDR)			_mm256_loadu_ps((ADDR))
#endif

static inline void *memset_sse2(void *ptr, int value, size_t num)
{
	__m128i m128i_value;
	int steps, remainder_step;
	int i;
	unsigned char *p_mov;

	m128i_value = _mm_set1_epi8((char)value);

	steps = (int)(num / sizeof(__m128i));
	remainder_step = num % sizeof(__m128i);

	p_mov = (unsigned char*)ptr;

	for (i = 0; i < steps; i++) {
		_mm_storeu_si128((__m128i*)p_mov, m128i_value);
		p_mov += sizeof(__m128i);
	}

	for (i = 0; i < remainder_step; i++)
		p_mov[i] = (char)value;

	return ptr;
}/*memset_sse2*/

#if(0) // not better than memset_sse2
static inline void *memset_avx(void *ptr, int value, size_t num)
{
	__m256i m256i_value;
	int steps, remainder_step;
	int i;
	unsigned char *p_mov;

	m256i_value = _mm256_set1_epi8((char)value);

	steps = (int)(num / sizeof(__m256i));
	remainder_step = num % sizeof(__m256i);

	p_mov = (unsigned char*)ptr;

	for (i = 0; i < steps; i++) {
		_mm256_store_si256((__m256i*)p_mov, m256i_value);
		p_mov += sizeof(m256i_value);
	}

	for (i = 0; i < remainder_step; i++)
		p_mov[i] = (char)value;

	return ptr;
}/*memset_avx*/
#endif




int SeparateConvolutionRowSerial(int width, int height, float const *p_extended_input,
	int kernel_length, float const *p_kernel_row,
	float *p_row_done_extended_output)
{
	int i, j;
	int kernel_radius;

	int extended_width;

	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */

	if (NULL == p_extended_input
		|| NULL == p_kernel_row
		|| NULL == p_row_done_extended_output)
	{
		return -3;
	}

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;

#if(1)
	memset(p_row_done_extended_output, 0,
		extended_width * height * sizeof(float));

	for (j = 0; j < height; j++) {
		int jj;

		int y_mul_input_width;

		y_mul_input_width = j*extended_width;

		for (jj = 0; jj < kernel_length; jj++) {

			int x;
			float kernel_element;

			x = kernel_radius;
			kernel_element = p_kernel_row[jj];

			for (i = 0; i < width; i++)
			{
				float product;
				product = kernel_element
					* p_extended_input[y_mul_input_width + x];

				p_row_done_extended_output[j*extended_width + x]
					+= product;
				x += 1;

			}/*for width*/

			y_mul_input_width += extended_width;
		}/*for kernel*/

	}/*for j*/
#else
	for (j = 0; j < height; j++) {
		int x;

		x = kernel_radius;
		
		memset(p_row_done_extended_output +
			j*extended_width, 0, kernel_radius * sizeof(float));

		for (i = 0; i < width; i++) {

			int jj;
			int y_mul_input_width;
			float sum;

			sum = 0;
			y_mul_input_width = j*extended_width;

			for (jj = 0; jj < kernel_length; jj++) {

				sum += p_kernel_row[jj]
					* p_extended_input[y_mul_input_width + x];

				y_mul_input_width += extended_width;
			}/*for kernel*/

			p_row_done_extended_output[j*extended_width + x] 
				= sum;
			x += 1;
		}/*for width*/

		memset(p_row_done_extended_output +
			j*extended_width + (kernel_radius + width), 0, 
			kernel_radius * sizeof(float));

	}/*for j*/
#endif

	return 0;

}/*SeparateConvolutionRowSerial*/


#ifdef _SWAP_KERNEL_AND_WIDTH

int SeparateConvolutionColumnSerial(int width, int height,
	float const *p_row_done_extended_input,
	int kernel_length, float const *p_kernel_column,
	float *p_output)
{
	int i, j;
	int kernel_radius;
	int extended_width;


	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */


	if (NULL == p_row_done_extended_input
		|| NULL == p_kernel_column
		|| NULL == p_output)
	{
		return -3;
	}

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;

	memset(p_output, 0, width*height*sizeof(float));

	{
		int y_mul_input_width;
		y_mul_input_width = 0;

		for (j = 0; j < height; j++) {
			int ii;
			for (ii = 0; ii < kernel_length; ii++) {
				
				int x;
				float kernel_element;

				x = ii;
				kernel_element = p_kernel_column[ii];

				for (i = 0; i < width; i++) {

					float product;

					product = kernel_element
						* p_row_done_extended_input[y_mul_input_width + x];
					p_output[j*width + i] += product;
					x += 1;
				}/*for width*/
			}/*kernel*/
			
			y_mul_input_width += extended_width;
		}/*for j*/
	}/*local variable*/

	return 0;
}/* for SeparateConvolutionColumnSerial*/

#else

int SeparateConvolutionColumnSerial(int width, int height,
	float const *p_row_done_extended_input,
	int kernel_length, float const *p_kernel_column,
	float *p_output)
{
	int i, j;
	int kernel_radius;
	int extended_width;


	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */


	if (NULL == p_row_done_extended_input
		|| NULL == p_kernel_column
		|| NULL == p_output)
	{
		return -3;
	}

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;

	{
		int y_mul_input_width;
		y_mul_input_width = 0;

		for (j = 0; j < height; j++) {

			for (i = 0; i < width; i++) {
				int x;
				int ii;
				float sum;

				x = i;
				sum = 0;

				for (ii = 0; ii < kernel_length; ii++) {

					sum += p_kernel_column[ii]
						* p_row_done_extended_input[y_mul_input_width + x];
					x += 1;
				}/*for kernel_length*/

				p_output[j*width + i] = sum;
			}/*for width*/

			y_mul_input_width += extended_width;
		}/*for j*/
	}

	return 0;
}/* for SeparateConvolutionColumnSerial*/

#endif

int SeparateConvolutionRowSSE4(int width, int height, 
	float const *p_extended_input,
	int kernel_length, float const *p_kernel_row,
	float *p_row_done_extended_output)
{
	int i, j;
	int kernel_radius;

	int extended_width;

	int steps, step_size;

	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */

	if (NULL == p_extended_input
		|| NULL == p_kernel_row
		|| NULL == p_row_done_extended_output)
	{
		return -3;
	}

	step_size = sizeof(__m128) / sizeof(float);
	steps = width / step_size;

	if (0 == steps)
	{
		SeparateConvolutionRowSerial(width, height,
			p_extended_input, kernel_length, p_kernel_row,
			p_row_done_extended_output);
	}/*if width < step_size*/

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;


	memset_sse2(p_row_done_extended_output, 0,
		extended_width * height * sizeof(float));

	for (j = 0; j < height; j++) {
		int jj;

		int y_mul_input_width;

		y_mul_input_width = j*extended_width;

		for (jj = 0; jj < kernel_length; jj++) {

			float *p_mov_extended_input;
			float *p_mov_output;

			float kernel_element;
			__m128 m128_kernel_element;

			int x;
			x = kernel_radius;

			kernel_element = p_kernel_row[jj];
			m128_kernel_element = _mm_set_ps1(kernel_element);

			p_mov_extended_input = 
				(float*)p_extended_input + y_mul_input_width + x;

			p_mov_output = 
				(float*)p_row_done_extended_output + j*extended_width + x;

			for (i = 0; i < steps; i++) {
				__m128 m_temp0, m_temp1, m_temp2, m_temp3;

				m_temp0 = _mm_loadu_ps(p_mov_extended_input);
				m_temp1 = _mm_mul_ps(m_temp0, m128_kernel_element);

				m_temp2 = _mm_loadu_ps(p_mov_output);

				m_temp3 = _mm_add_ps(m_temp1, m_temp2);

				_mm_storeu_ps(p_mov_output, m_temp3);

				p_mov_extended_input += step_size;
				p_mov_output += step_size;
			}/*for sse*/

			for (i = steps*step_size; i < width; i++) {
				p_mov_output[0] += 
					p_mov_extended_input[0] * kernel_element;
				p_mov_extended_input += 1;
				p_mov_output += 1;
			}/*remainder*/

			y_mul_input_width += extended_width;
		}/*for kernel*/

	}/*for j*/

	return 0;
}/*SeparateConvolutionRowSSE4*/


#ifdef _SWAP_KERNEL_AND_WIDTH

int SeparateConvolutionColumnSSE4(int width, int height,
	float const *p_row_done_extended_input,
	int kernel_length, float const *p_kernel_column,
	float *p_output)
{

	int i, j;
	int kernel_radius;
	int extended_width;

	int step_size;
	int steps;


	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */


	if (NULL == p_row_done_extended_input
		|| NULL == p_kernel_column
		|| NULL == p_output)
	{
		return -3;
	}

	step_size = sizeof(__m128) / sizeof(float);
	steps = width / step_size;

	if (0 == steps)
	{
		return SeparateConvolutionColumnSerial(width, height,
			p_row_done_extended_input,
			kernel_length, p_kernel_column, p_output);
	}/*if */

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;

	memset_sse2(p_output, 0, width*height*sizeof(float));

	{
		int y_mul_input_width;
		y_mul_input_width = 0;

		for (j = 0; j < height; j++) {
			int ii;
			for (ii = 0; ii < kernel_length; ii++) {

				float kernel_element;
				__m128 m128_kernel_element;

				float *p_mov_extended_input;
				float *p_mov_output;
				int x;

				kernel_element = p_kernel_column[ii];
				m128_kernel_element = _mm_set_ps1(kernel_element);

				x = ii;
				p_mov_extended_input =
					(float*)p_row_done_extended_input + y_mul_input_width + x;
				p_mov_output = (float*)p_output + j*width;


				for (i = 0; i < steps; i++) {
					__m128 m_temp0, m_temp1, m_temp2, m_temp3;

					m_temp0 = _mm_loadu_ps(p_mov_extended_input);
					m_temp1 = _mm_mul_ps(m_temp0, m128_kernel_element);

					m_temp2 = _mm_loadu_ps(p_mov_output);
					m_temp3 = _mm_add_ps(m_temp1, m_temp2);

					_mm_storeu_ps(p_mov_output, m_temp3);

					p_mov_extended_input += step_size;
					p_mov_output += step_size;
				}/*for sse*/

				for (i = steps*step_size; i < steps; i++) {
					float product;

					product = kernel_element
						* p_mov_extended_input[0];

					p_mov_output[0] += product;
					p_mov_extended_input += 1;
					p_mov_output += 1;
				}/*for remainder*/

			}/*for kernel*/

			y_mul_input_width += extended_width;
		}/*for j*/

	}/*local variable*/

	return 0;
}/*SeparateConvolutionColumnSSE4*/

#else

int SeparateConvolutionColumnSSE4(int width, int height,
	float const *p_row_done_extended_input,
	int kernel_length, float const *p_kernel_column,
	float *p_output)
{

	int i, j;
	int kernel_radius;
	int extended_width;

	int step_size;
	int steps;


	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */


	if (NULL == p_row_done_extended_input
		|| NULL == p_kernel_column
		|| NULL == p_output)
	{
		return -3;
	}

	step_size = sizeof(__m128) / sizeof(float);
	steps = kernel_length / step_size;

	if (0 == steps)
	{
		return SeparateConvolutionColumnSerial(width, height, 
			p_row_done_extended_input,
			kernel_length, p_kernel_column, p_output);
	}/*if */

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;

	{
		int y_mul_input_width;
		y_mul_input_width = 0;

		for (j = 0; j < height; j++) {

			for (i = 0; i < width; i++) {
				int x;
				int ii;

				float *p_mov_kernel;
				float *p_mov_row_done_extended_input;

				float sum;

				sum = 0;
				x = i;

				p_mov_kernel = (float*)p_kernel_column;
				p_mov_row_done_extended_input 
					= (float*)p_row_done_extended_input + y_mul_input_width + x;

				for (ii = 0; ii < steps; ii++) {
					__m128 m_kernel, m_input;
					__m128 m_temp0;

					float temp_sum;

					m_kernel = __MM128_LOAD_KERNEL(p_mov_kernel);
					m_input = _mm_loadu_ps(p_mov_row_done_extended_input);


					m_temp0 = _mm_dp_ps(m_kernel, m_input, 0xf1);
					temp_sum = _mm_cvtss_f32(m_temp0);

					sum += temp_sum;

					p_mov_kernel += step_size;
					p_mov_row_done_extended_input += step_size;
				}/*for sse*/

				for (ii = steps*step_size; ii < kernel_length; ii++) {
					sum += p_mov_kernel[0] * p_mov_row_done_extended_input[0];
					p_mov_kernel += 1;
					p_mov_row_done_extended_input += 1;
				}/*for remainder*/

				p_output[j*width + i] = sum;
			}/*for width*/

			y_mul_input_width += extended_width;
		}/*for j*/

	}/*local variable*/

	return 0;
}/*SeparateConvolutionColumnSSE4*/

#endif


int SeparateConvolutionRowAVX(int width, int height, 
	float const *p_extended_input,
	int kernel_length, float const *p_kernel_row,
	float *p_row_done_extended_output)
{
	int i, j;
	int kernel_radius;

	int extended_width;

	int steps_avx, step_size_avx;
	int remainder_avx;

	int steps_sse, step_size_sse;
	int remainder_sse;

	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */

	if (NULL == p_extended_input
		|| NULL == p_kernel_row
		|| NULL == p_row_done_extended_output)
	{
		return -3;
	}

	step_size_avx = sizeof(__m256) / sizeof(float);
	steps_avx = width / step_size_avx;
	remainder_avx = width % steps_avx;

	if (0 == steps_avx)
	{
		SeparateConvolutionRowSSE4(width, height,
			p_extended_input, kernel_length, p_kernel_row,
			p_row_done_extended_output);
	}/*if width < step_size*/

	step_size_sse = sizeof(__m128) / sizeof(float);
	steps_sse = remainder_avx / step_size_sse;

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;

	memset_sse2(p_row_done_extended_output, 0,
		extended_width * height * sizeof(float));

	for (j = 0; j < height; j++) {
		int jj;

		int y_mul_input_width;

		y_mul_input_width = j*extended_width;

		for (jj = 0; jj < kernel_length; jj++) {

			float *p_mov_extended_input;
			float *p_mov_output;

			float kernel_element;
			__m256 m256_kernel_element;
			__m128 m128_kernel_element;

			int x;
			x = kernel_radius;

			kernel_element = p_kernel_row[jj];

			m256_kernel_element = _mm256_set1_ps(kernel_element);
			m128_kernel_element = _mm_set_ps1(kernel_element);

			p_mov_extended_input =
				(float*)p_extended_input + y_mul_input_width + x;

			p_mov_output =
				(float*)p_row_done_extended_output + j*extended_width + x;

			for (i = 0; i < steps_avx; i++) {

				__m256 m256_temp0, m256_temp1, m256_temp2, m256_temp3;

				m256_temp0 = _mm256_loadu_ps(p_mov_extended_input);
				m256_temp1 = _mm256_mul_ps(m256_temp0, m256_kernel_element);

				m256_temp2 = _mm256_loadu_ps(p_mov_output);
				m256_temp3 = _mm256_add_ps(m256_temp1, m256_temp2);
				_mm256_storeu_ps(p_mov_output, m256_temp3);

				p_mov_extended_input += step_size_avx;
				p_mov_output += step_size_avx;
			}/*for avx*/

			for (i = 0; i < steps_sse; i++) {
				__m128 m128_temp0, m128_temp1, m128_temp2, m128_temp3;

				m128_temp0 = _mm_loadu_ps(p_mov_extended_input);
				m128_temp1 = _mm_mul_ps(m128_temp0, m128_kernel_element);

				m128_temp2 = _mm_loadu_ps(p_mov_output);
				m128_temp3 = _mm_add_ps(m128_temp1, m128_temp2);

				_mm_storeu_ps(p_mov_output, m128_temp3);

				p_mov_extended_input += step_size_sse;
				p_mov_output += step_size_sse;
			}/*for sse*/

		
			i = steps_avx*step_size_avx
				+ step_size_sse * steps_sse;

			for (; i < width; i++) {
				p_mov_output[0] +=
					p_mov_extended_input[0] * kernel_element;
				p_mov_extended_input += 1;
				p_mov_output += 1;
			}/*remainder */

			y_mul_input_width += extended_width;
		}/*for kernel*/

	}/*for j*/

	return 0;
}/*SeparateConvolutionRowAVX*/

#ifdef _SWAP_KERNEL_AND_WIDTH

int SeparateConvolutionColumnAVX(int width, int height,
	float const *p_row_done_extended_input,
	int kernel_length, float const *p_kernel_column,
	float *p_output)
{
	int i, j;
	int kernel_radius;
	int extended_width;

	int step_size_avx, steps_avx;
	int remainder_avx;

	int step_size_sse, steps_sse;
	int remainder_sse;

	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */


	if (NULL == p_row_done_extended_input
		|| NULL == p_kernel_column
		|| NULL == p_output)
	{
		return -3;
	}

	step_size_avx = sizeof(__m256) / sizeof(float);
	steps_avx = width / step_size_avx;
	remainder_avx = width % step_size_avx;

	if (0 == steps_avx)
	{
		return SeparateConvolutionColumnSSE4(width, height,
			p_row_done_extended_input,
			kernel_length, p_kernel_column, p_output);
	}/*if */

	step_size_sse = sizeof(__m128) / sizeof(float);
	steps_sse = remainder_avx / step_size_sse;
	remainder_sse = remainder_avx % step_size_sse;

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;

	memset_sse2(p_output, 0, width*height * sizeof(float));

	{
		int y_mul_input_width;
		y_mul_input_width = 0;

		for (j = 0; j < height; j++) {
			int ii;

			for (ii = 0; ii < kernel_length; ii++) {

				float *p_mov_extended_input;
				float *p_mov_output;

				float kernel_element;
				__m256 m256_kernel_element;
				__m128 m128_kernel_element;

				int x;

				kernel_element = p_kernel_column[ii];

				m256_kernel_element = _mm256_set1_ps(kernel_element);
				m128_kernel_element = _mm_set_ps1(kernel_element);

				x = ii;
				p_mov_extended_input =
					(float*)p_row_done_extended_input + y_mul_input_width + x;				
				p_mov_output = (float*)p_output + j*width;

				for (i = 0; i < steps_avx; i++) {

					__m256 m256_temp0, m256_temp1, m256_temp2, m256_temp3;

					m256_temp0 = _mm256_loadu_ps(p_mov_extended_input);
					m256_temp1 = _mm256_mul_ps(m256_temp0, m256_kernel_element);

					m256_temp2 = _mm256_loadu_ps(p_mov_output);
					m256_temp3 = _mm256_add_ps(m256_temp1, m256_temp2);

					_mm256_storeu_ps(p_mov_output, m256_temp3);

					p_mov_extended_input += step_size_avx;
					p_mov_output += step_size_avx;
				}/*for avx*/

				for (i = 0; i < steps_sse; i++) {
					__m128 m128_temp0, m128_temp1, m128_temp2, m128_temp3;

					m128_temp0 = _mm_loadu_ps(p_mov_extended_input);
					m128_temp1 = _mm_mul_ps(m128_temp0, m128_kernel_element);

					m128_temp2 = _mm_loadu_ps(p_mov_output);
					m128_temp3 = _mm_add_ps(m128_temp1, m128_temp2);

					_mm_storeu_ps(p_mov_output, m128_temp3);

					p_mov_extended_input += step_size_sse;
					p_mov_output += step_size_sse;
				}/*for sse*/


				i = steps_avx*step_size_avx
					+ step_size_sse * steps_sse;

				for (; i < width; i++) {
					float product;

					product = kernel_element
						* p_mov_extended_input[0];

					p_mov_output[0] += product;
					p_mov_extended_input += 1;
					p_mov_output += 1;
				}/*remainder*/

			}/*for kernel*/

			y_mul_input_width += extended_width;
		}/*for j*/
	}
	return 0;
}/*SeparateConvolutionColumnAVX*/

#else

int SeparateConvolutionColumnAVX(int width, int height,
	float const *p_row_done_extended_input,
	int kernel_length, float const *p_kernel_column,
	float *p_output)
{

	int i, j;
	int kernel_radius;
	int extended_width;

	int step_size_avx, steps_avx;
	int remainder_avx;

	int step_size_sse, steps_sse;
	int remainder_sse;

	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */


	if (NULL == p_row_done_extended_input
		|| NULL == p_kernel_column
		|| NULL == p_output)
	{
		return -3;
	}

	step_size_avx = sizeof(__m256) / sizeof(float);
	steps_avx = kernel_length / step_size_avx;
	remainder_avx = kernel_length % step_size_avx;

	if (0 == steps_avx)
	{
		return SeparateConvolutionColumnSSE4(width, height,
			p_row_done_extended_input,
			kernel_length, p_kernel_column, p_output);
	}/*if */


	step_size_sse = sizeof(__m128) / sizeof(float);
	steps_sse = remainder_avx / step_size_sse;
	remainder_sse = remainder_avx % step_size_sse;

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;

	{
		int y_mul_input_width;
		y_mul_input_width = 0;

		for (j = 0; j < height; j++) {

			for (i = 0; i < width; i++) {
				int x;
				int ii;

				float *p_mov_kernel;
				float *p_mov_row_done_extended_input;

				float sum;

				sum = 0;
				x = i;

				p_mov_kernel = (float*)p_kernel_column;
				p_mov_row_done_extended_input
					= (float*)p_row_done_extended_input + y_mul_input_width + x;

				for (ii = 0; ii < steps_avx; ii++) {

					__m256 m256_kernel, m256_src;
					__m256 m256_temp0;

					float temp_sum;

					m256_kernel = __MM256_LOAD_KERNEL(p_mov_kernel);
					m256_src = _mm256_loadu_ps(p_mov_row_done_extended_input);

					{
						__m128  m128_temp0, m128_temp1, m128_temp2, m128_temp3,
							m128_temp4, m128_temp5, m128_temp6;

						m256_temp0 = _mm256_mul_ps(m256_kernel, m256_src);
						m128_temp0 = _mm256_castps256_ps128(m256_temp0);
						m128_temp1 = _mm256_extractf128_ps(m256_temp0, 1);

						m128_temp2 = _mm_add_ps(m128_temp0, m128_temp1);

#if(0)					// linker or illegal instruction crash
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
					}/*local variables*/

					sum += temp_sum;

					p_mov_kernel += step_size_avx;
					p_mov_row_done_extended_input += step_size_avx;
				}/*for avx*/

				for (ii = 0; ii < steps_sse; ii++) {
					__m128 m128_kernel, m128_input;
					__m128 m128_temp0;

					float temp_sum;

					m128_kernel = __MM128_LOAD_KERNEL(p_mov_kernel);
					m128_input = _mm_loadu_ps(p_mov_row_done_extended_input);


					m128_temp0 = _mm_dp_ps(m128_kernel, m128_input, 0xf1);
					temp_sum = _mm_cvtss_f32(m128_temp0);

					sum += temp_sum;

					p_mov_kernel += step_size_sse;
					p_mov_row_done_extended_input += step_size_sse;
				}/*for sse*/

				ii = steps_avx*step_size_avx
					+ step_size_sse * steps_sse;

				for (; ii < kernel_length; ii++) {
					sum += p_mov_kernel[0] * p_mov_row_done_extended_input[0];
					p_mov_kernel += 1;
					p_mov_row_done_extended_input += 1;
				}/*for remainder*/

				p_output[j*width + i] = sum;
			}/*width*/

			y_mul_input_width += extended_width;
		}/*kernel*/

	}/*local variable*/

	return 0;
}/* for SeparateConvolutionColumnSSE4*/
#endif