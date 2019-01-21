#include <stdlib.h>


#pragma warning(disable:4752)
#include <immintrin.h>

static void *memset_sse2(void *ptr, int value, size_t num)
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
static void *memset_avx(void *ptr, int value, size_t num)
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
	int kernel_length, float const *p_kernel_column, 
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
		|| NULL == p_kernel_column
		|| NULL == p_row_done_extended_output)
	{
		return -3;
	}

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;

	for (j = 0; j < kernel_radius; j++) {
		memset(p_row_done_extended_output + j*width, 0,
			width * sizeof(float));
	}/*for j*/

	for (j = 0; j < height; j++) {

		int y_mul_input_width;
		y_mul_input_width = (kernel_radius + j)*extended_width;

		for (i = 0; i < width; i++) {

			int ii;
			float sum;
			sum = 0;

			for (ii = 0; ii < kernel_length; ii++) {

				int x;
				x = i + ii;

				sum	+= p_kernel_column[ii]
					*p_extended_input[y_mul_input_width + x];
				
			}/*for */

			p_row_done_extended_output[(kernel_radius + j)*width + i] = sum;
		}/*for i*/

	}/*for j*/


	for (j = 0; j < kernel_radius; j++) {
		memset(p_row_done_extended_output + (j + kernel_radius + height)*width, 0,
			width * sizeof(float));
	}/*for j*/

	return 0;
}/*SeparateConvolutionRowSerial*/


int SeparateConvolutionColumnSerial(int width, int height, 
	float const *p_row_done_extended_input,
	int kernel_length, float const *p_kernel_row, 
	float *p_output)
{
	int i, j;

	
	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */


	if (NULL == p_row_done_extended_input
		|| NULL == p_kernel_row
		|| NULL == p_output)
	{
		return -3;
	}


#if(0)
	for (j = 0; j < height; j++) {

		for (i = 0; i < width; i++) {

			int jj;
			float sum;

			sum = 0;

			for (jj = 0; jj < kernel_length; jj++) {
				int x;
				int y_mul_input_width;

				x = i;
				y_mul_input_width = (j + jj)*width;

				sum += p_kernel_row[jj]
					* p_row_done_extended_input[y_mul_input_width + x];
			}/*for */

			p_output[j*width + i] = sum;
		}/*for i*/

	}/*for j*/
#else
	{
		int jj;
	
		memset(p_output, 0, width*height*sizeof(float));
		for (j = 0; j < height; j++) {
			for (jj = 0; jj < kernel_length; jj++) {
				float kernel_element;
				kernel_element = p_kernel_row[jj];

				for (i = 0; i < width; i++) {
					int x;
					int y_mul_input_width;
				
					x = i;
					y_mul_input_width = (j + jj)*width;

					p_output[j*width + i] += kernel_element
						* p_row_done_extended_input[y_mul_input_width + x];

				}/*i*/
			}/*jj*/
		}/*for j*/	
	}/*local variable*/
#endif

	return 0;
}/*SeparateConvolutionColumnSerial*/


int SeparateConvolutionRowSSE4(int width, int height, float const *p_extended_input,
	int kernel_length, float const *p_kernel_column, float 
	*p_row_done_extended_output)
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


	if (NULL == p_extended_input
		|| NULL == p_kernel_column
		|| NULL == p_row_done_extended_output)
	{
		return -3;
	}

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;

	step_size = sizeof(__m128) / sizeof(float);
	steps = kernel_length / step_size;

	if (kernel_length < step_size)
	{
		return SeparateConvolutionRowSerial(width, height, p_extended_input,
			kernel_length, p_kernel_column, p_row_done_extended_output);
	}/*if */


	for (j = 0; j < kernel_radius; j++) {
		memset_sse2(p_row_done_extended_output + j*width, 0,
			width * sizeof(float));
	}/*for j*/

	for (j = 0; j < height; j++) {

		int y_mul_input_width;
		y_mul_input_width = (kernel_radius + j)*extended_width;

		for (i = 0; i < width; i++) {

			int ii;
			float sum;

			float *p_mov_input;
			float *p_mov_kernel;
			int x;

			sum = 0;
			x = i;
			p_mov_kernel = (float*)p_kernel_column;
			p_mov_input = (float*)p_extended_input + y_mul_input_width + x;


			for (ii = 0; ii < steps; ii++) {
				__m128 m_kernel, m_input;
				__m128 m_temp0;

				float temp_sum;

				m_kernel = _mm_loadu_ps(p_mov_kernel);
				m_input = _mm_loadu_ps(p_mov_input);


				m_temp0 = _mm_dp_ps(m_kernel, m_input, 0xf1);
				temp_sum = _mm_cvtss_f32(m_temp0);

				sum += temp_sum;

				p_mov_kernel += step_size;
				p_mov_input += step_size;
			}/*for steps*/

			for (ii = steps*step_size; ii < kernel_length; ii++) {
				sum += p_mov_kernel[0] * p_mov_input[0];
				p_mov_kernel += 1;
				p_mov_input += 1;
			}/*for remainder*/

			p_row_done_extended_output[(kernel_radius + j)*width + i] = sum;
		}/*for i*/

	}/*for j*/

	for (j = 0; j < kernel_radius; j++) {
		memset_sse2(p_row_done_extended_output + (j + kernel_radius + height)*width, 0,
			width * sizeof(float));
	}/*for j*/

	return 0;
}/*SeparateConvolutionRowSSE4*/


int SeparateConvolutionColumnSSE4(int width, int height,
	float const *p_row_done_extended_input,
	int kernel_length, float const *p_kernel_row, 
	float *p_output)
{
	int i, j;
	int jj;
	int steps, step_size;


	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */


	if (NULL == p_row_done_extended_input
		|| NULL == p_kernel_row
		|| NULL == p_output)
	{
		return -3;
	}

	step_size = sizeof(__m128) / sizeof(float);

	if (width < step_size)
	{
		SeparateConvolutionColumnSerial(width, height,
			p_row_done_extended_input, kernel_length, p_kernel_row, 
			p_output);
	}/*if width < step_size*/

	steps = width/step_size;

	memset_sse2(p_output, 0, width*height * sizeof(float));

	for (j = 0; j < height; j++) {
		for (jj = 0; jj < kernel_length; jj++) {

			float *p_mov_input;
			float *p_mov_output;
			float kernel_element;

			__m128 m128_kernel_element;
			int y_mul_input_width;

			kernel_element = p_kernel_row[jj];
			m128_kernel_element = _mm_set_ps1(kernel_element);

			y_mul_input_width = (j + jj)*width;

			p_mov_input = (float*)p_row_done_extended_input + y_mul_input_width;
			p_mov_output = (float*)p_output + j*width;

			for (i = 0; i < steps; i++) {
				__m128 m_temp0, m_temp1, m_temp2, m_temp3;

				m_temp0 = _mm_loadu_ps(p_mov_input);
				m_temp1 = _mm_mul_ps(m_temp0, m128_kernel_element);

				m_temp2 = _mm_loadu_ps(p_mov_output);
				m_temp3 = _mm_add_ps(m_temp1, m_temp2);

				_mm_storeu_ps(p_mov_output, m_temp3);

				p_mov_input += step_size;
				p_mov_output += step_size;
			}/*for i*/

			for (i = steps*step_size; i < width; i++) {
				p_mov_output[0] += p_mov_input[0] * kernel_element;
				p_mov_input += 1;
				p_mov_output += 1;
			}/*remainder of 4*/

		}/*jj*/

	}/*for j*/

	return 0;
}/*SeparateConvolutionColumnSSE4*/


int SeparateConvolutionRowAVX(int width, int height, float const *p_extended_input,
	int kernel_length, float const *p_kernel_column, 
	float *p_row_done_extended_output)
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


	if (NULL == p_extended_input
		|| NULL == p_kernel_column
		|| NULL == p_row_done_extended_output)
	{
		return -3;
	}

	kernel_radius = kernel_length / 2;
	extended_width = width + 2 * kernel_radius;

	step_size_avx = sizeof(__m256) / sizeof(float);

	if (kernel_length < step_size_avx)
	{
		return SeparateConvolutionRowSerial(width, height, p_extended_input,
			kernel_length, p_kernel_column, p_row_done_extended_output);
	}/*if */

	steps_avx = kernel_length / step_size_avx;
	remainder_avx = kernel_length % step_size_avx;

	step_size_sse = sizeof(__m128) / sizeof(float);
	steps_sse = remainder_avx / step_size_sse;
	remainder_sse = remainder_avx % step_size_sse;


	for (j = 0; j < kernel_radius; j++) {		
		memset_sse2(p_row_done_extended_output + j*width, 0,
				width * sizeof(float));
	}/*for j*/

	for (j = 0; j < height; j++) {

		int y_mul_input_width;
		y_mul_input_width = (kernel_radius + j)*extended_width;

		for (i = 0; i < width; i++) {

			int ii;
			float sum;

			float *p_mov_input;
			float *p_mov_kernel;
			int x;

			sum = 0;
			x = i;
			p_mov_kernel = (float*)p_kernel_column;
			p_mov_input = (float*)p_extended_input + y_mul_input_width + x;

			for (ii = 0; ii < steps_avx; ii++) {

				__m256 m256_kernel, m256_src;
				__m256 m256_temp0;

				float temp_sum;


				m256_kernel = _mm256_loadu_ps(p_mov_kernel);
				m256_src = _mm256_loadu_ps(p_mov_input);


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
				}

				sum += temp_sum;

				p_mov_kernel += step_size_avx;
				p_mov_input += step_size_avx;
			}/*for avx*/

			for (ii = 0; ii < steps_sse; ii++) {
				__m128 m_kernel, m_input;
				__m128 m_temp0;

				float temp_sum;

				m_kernel = _mm_loadu_ps(p_mov_kernel);
				m_input = _mm_loadu_ps(p_mov_input);

				m_temp0 = _mm_dp_ps(m_kernel, m_input, 0xf1);
				temp_sum = _mm_cvtss_f32(m_temp0);

				sum += temp_sum;

				p_mov_kernel += step_size_sse;
				p_mov_input += step_size_sse;
			}/*for sse*/

			{
				int serial_begin;
				serial_begin = steps_avx*step_size_avx
					+ step_size_sse * steps_sse;

				for (ii = serial_begin; ii < kernel_length; ii++) {
					sum += p_mov_kernel[0] * p_mov_input[0];
					p_mov_kernel += 1;
					p_mov_input += 1;
				}/*for ii*/
			}/* remainder of 4 */

			p_row_done_extended_output[(kernel_radius + j)*width + i] = sum;
		}/*for i*/

	}/*for j*/

	for (j = 0; j < kernel_radius; j++) {		
		memset_sse2(p_row_done_extended_output + (j + kernel_radius + height)*width, 0,
			width * sizeof(float));
	}/*for j*/

	return 0;
}/*SeparateConvolutionRowAVX*/



int SeparateConvolutionColumnAVX(int width, int height,
	float const *p_row_done_extended_input,
	int kernel_length, float const *p_kernel_row,
	float *p_output)
{
	int i, j;
	int jj;

	int step_size_avx, steps_avx;
	int remainder_avx;

	int steps_sse, step_size_sse;
	int remainder_sse;

	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */


	if (NULL == p_row_done_extended_input
		|| NULL == p_kernel_row
		|| NULL == p_output)
	{
		return -3;
	}

	step_size_avx = sizeof(__m256) / sizeof(float);

	if (width < step_size_avx)
	{
		SeparateConvolutionColumnSSE4(width, height,
			p_row_done_extended_input, kernel_length, p_kernel_row,
			p_output);
	}/*if width < step_size*/

	steps_avx = width / step_size_avx;
	remainder_avx = width % steps_avx;

	step_size_sse = sizeof(__m128) / sizeof(float);
	steps_sse = remainder_avx / step_size_sse;
	remainder_sse = remainder_avx % step_size_sse;

	memset_sse2(p_output, 0, width*height * sizeof(float));

	for (j = 0; j < height; j++) {
		for (jj = 0; jj < kernel_length; jj++) {

			float *p_mov_input;
			float *p_mov_output;
			float kernel_element;

			__m256 m256_kernel_element;
			__m128 m128_kernel_element;

			int y_mul_input_width;

			kernel_element = p_kernel_row[jj];

			m256_kernel_element = _mm256_set1_ps(kernel_element);
			m128_kernel_element = _mm_set_ps1(kernel_element);

			y_mul_input_width = (j + jj)*width;

			p_mov_input = (float*)p_row_done_extended_input + y_mul_input_width;
			p_mov_output = (float*)p_output + j*width;

			for (i = 0; i < steps_avx; i++) {
				__m256 m256_temp0, m256_temp1, m256_temp2, m256_temp3;

				m256_temp0 = _mm256_loadu_ps(p_mov_input);
				m256_temp1 = _mm256_mul_ps(m256_temp0, m256_kernel_element);

				m256_temp2 = _mm256_loadu_ps(p_mov_output);
				m256_temp3 = _mm256_add_ps(m256_temp1, m256_temp2);
				_mm256_storeu_ps(p_mov_output, m256_temp3);

				p_mov_input += step_size_avx;
				p_mov_output += step_size_avx;
			}/*avx*/

			for (i = 0; i < steps_sse; i++) {
				__m128 m128_temp0, m128_temp1, m128_temp2, m128_temp3;

				m128_temp0 = _mm_loadu_ps(p_mov_input);
				m128_temp1 = _mm_mul_ps(m128_temp0, m128_kernel_element);

				m128_temp2 = _mm_loadu_ps(p_mov_output);
				m128_temp3 = _mm_add_ps(m128_temp1, m128_temp2);

				_mm_storeu_ps(p_mov_output, m128_temp3);

				p_mov_input += step_size_sse;
				p_mov_output += step_size_sse;
			}/*for sse*/

			{
				int serial_begin;
				serial_begin = steps_avx*step_size_avx
					+ step_size_sse * steps_sse;

				for (i = serial_begin; i < width; i++) {
					p_mov_output[0] += p_mov_input[0] * kernel_element;
					p_mov_input += 1;
					p_mov_output += 1;
				}/*remainder of 4*/
			}

		}/*jj*/

	}/*for j*/

	return 0;
}/*SeparateConvolutionColumnAVX*/