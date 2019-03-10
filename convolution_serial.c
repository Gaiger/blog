
#include <stdlib.h>

#include "convolution_serial.h"

int ConvolutionSerialCPU(int width, int height, float *p_input,
	int kernel_length, float *p_kernel,
	float *p_output)
{
	int i, j;
	int kernel_radius;

	if (0 == width || 0 == height)
		return -1;

	if (kernel_length > width || kernel_length > height)
	{
		return -2;
	}/*if */

	if (NULL == p_input
		|| NULL == p_kernel
		|| NULL == p_output)
	{
		return -3;
	}

	kernel_radius = kernel_length / 2;


	for (j = 0; j < height; j++) {
		for (i = 0; i < width; i++) {

			int ii, jj;
			float sum;
			sum = 0;

			for (jj = 0; jj < kernel_length; jj++) {
				for (ii = 0; ii < kernel_length; ii++) {
					int x, y;

					x = i + ii - kernel_radius;
					y = j + jj - kernel_radius;


					if ((x < 0 || x >= width)
						|| (y < 0 || y >= height))
					{
						continue;
					}/*out of bound*/

					sum += p_kernel[kernel_length*jj + ii]
						* p_input[y*width + x];
				}/*ii*/
			}/*jj*/

			p_output[j*width + i] = sum;
		}/*for i*/
	}/*for j*/

	return 0;
}/*ConvolutionSerialCPU*/


int ConvolutionSerialExtensionCPU(int width, int height,
	float *p_extended_input, int kernel_length, float *p_kernel,
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
		|| NULL == p_kernel
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
				for (ii = 0; ii < kernel_length; ii++) {
					int x, y;
					x = i + ii;
					y = j + jj;


					sum += p_kernel[kernel_length*jj + ii]
						* p_extended_input[extended_width*y + x];
				}/*for ii*/
			}/*for jj*/

			p_output[j*width + i] = sum;
		}/*for i*/
	}/*for j*/

	return 0;

}/*ConvolutionSerialExternionCPU*/
