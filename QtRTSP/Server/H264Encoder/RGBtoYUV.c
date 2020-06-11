#include <stdlib.h>

#include "RGBtoYUV.h"


#define LOCAL					static

#define DIV2(VAL)				((VAL)>>1)
#define DIV4(VAL)				((VAL)>>2)
#define DIV8(VAL)				((VAL)>>3)
#define DIV128(VAL)				((VAL)>>7)

#define MUL2(VAL)				((VAL)<<1)
#define MUL4(VAL)				((VAL)<<2)
#define MUL8(VAL)				((VAL)<<3)


/*
	RGB to Y :
		Y = (R*38 + G*75 + B*15) / 128
*/

/*
  U V :
	BY to U:
		U = (1 / 1.772)*(B-Y)
	RY to V :
		V = (1 / 1.1402)*(R-Y)
*/

#define ALIGN_TO_FOUR(VAL)		(((VAL + 3)/4)*4)
//#define ALIGN_TO_FOUR(VAL)		(VAL)

#define RGB_TO_Y(RR,GG,BB)		(DIV128( 38 * (RR) + 75 * (GG) + 15 * (BB) ))
#define BY2U(BB, YY)			( ( (71)*( (BB) - (YY) )  >> 7) + 128)
#define RY2V(RR, YY)			( ( (91)*( (RR) - (YY) )  >> 7) + 128)


LOCAL int RGBtoY(unsigned char *p_RGB, int width, int height,
				 int width_step_Y, unsigned char *p_Y)
{
	int i, j;

	unsigned char *p_mov_RGB;
	unsigned char *p_line_RGB;

	unsigned char *p_mov_Y;
	unsigned char *p_line_Y;

	int width_step_RGB;

	width_step_RGB = ALIGN_TO_FOUR(3 * width_step_Y);

	p_line_RGB = p_RGB;
	p_line_Y = p_Y;

	for(j = 0; j< height; j++){

		p_mov_RGB = p_line_RGB;
		p_mov_Y = p_line_Y;

		for(i = 0; i< width; i++){
			int R, G, B;
			R = *(p_mov_RGB + 0);
			G = *(p_mov_RGB + 1);
			B = *(p_mov_RGB + 2);

			*p_mov_Y = RGB_TO_Y(R, G, B);

			p_mov_RGB += 3;
			p_mov_Y++;
		}/*for i */

		p_line_RGB += width_step_RGB;
		p_line_Y += width_step_Y;
	}/*for j*/

	return 0;
}

/**********************************************************************/

LOCAL int RGBtoU420(unsigned char *p_B, int width, int height,
			  unsigned  char *p_Y, int width_step_Y,  unsigned char *p_U)
{
	int i, j;
	unsigned char *p_mov_B;
	unsigned char *p_mov_Y, *p_mov_U;

	unsigned char *p_line_B;
	unsigned char *p_line_Y, *p_line_U;

	int double_width_RGB;
	int half_width, half_height;
	int line_step_Y, line_step_U;


	double_width_RGB = MUL2(ALIGN_TO_FOUR(width_step_Y * 3));

	half_width = DIV2(width);
	half_height = DIV2(height);

	line_step_Y = MUL2(width_step_Y);
	line_step_U = DIV2(width_step_Y);

	p_line_B = p_B;
	p_line_Y =p_Y;

	p_line_U = p_U;

	for(j = 0; j< half_height; j++){

		p_mov_B = p_line_B;
		p_mov_Y = p_line_Y;

		p_mov_U = p_line_U;

		for(i = 0; i< half_width; i++){

			*p_mov_U = BY2U( *((unsigned char*)p_mov_B), *p_mov_Y);

			p_mov_B += 6;
			p_mov_Y += 2;

			p_mov_U++;
		}/*for i */

		p_line_B += double_width_RGB;

		p_line_Y += line_step_Y;
		p_line_U += line_step_U;
	}/*for j*/

	return 0;
}

/**********************************************************************/

LOCAL int RGB32toV420(unsigned  char *pR, int width, int height,
				  unsigned  char *p_Y, int width_step_Y, unsigned char *p_V)
{
	int i, j;
	unsigned char *p_mov_R;
	unsigned char *p_mov_Y, *p_mov_V;

	unsigned char *p_line_R;
	unsigned char *p_line_Y, *p_line_V;

	int half_width, half_height;
	int double_width_RGB;
	int line_step_Y, VlineStep;

	double_width_RGB = MUL2(ALIGN_TO_FOUR(width_step_Y*3));
	half_width = DIV2(width);
	half_height = DIV2(height);

	line_step_Y = MUL2(width_step_Y);

	VlineStep = DIV2(width_step_Y);


	p_line_R = pR;
	p_line_Y = p_Y;

	p_line_V = p_V;

	for(j = 0; j< half_height; j++){

		p_mov_R = p_line_R;
		p_mov_Y = p_line_Y;

		p_mov_V = p_line_V;

		for(i = 0; i< half_width; i++){

			*p_mov_V = RY2V( *((unsigned char*)p_mov_R), *p_mov_Y);

			p_mov_R += 6;
			p_mov_Y += 2;

			p_mov_V++;
		}/*for i */

		p_line_R += double_width_RGB;
		p_line_Y += line_step_Y;

		p_line_V += VlineStep;
	}/*for j*/

	return 0;
}

/**********************************************************************/

LOCAL int RGBtoUV420Interleave(unsigned char *p_RGB, int width, int height,
						 unsigned char *p_Y, int width_step_Y, unsigned char *p_UV)
{
	int i, j;
	unsigned char *p_mov_RGB;
	unsigned char *p_mov_Y, *p_mov_UV;

	unsigned char *p_line_RGB;
	unsigned char *p_line_Y, *p_line_UV;

	int half_width, half_height;
	int double_width_RGB;
	int line_step_Y, line_Step_UV;

	double_width_RGB = MUL2(ALIGN_TO_FOUR(width_step_Y*3));
	half_width = DIV2(width);
	half_height = DIV2(height);

	line_step_Y = MUL2(width_step_Y);

	line_Step_UV = width_step_Y;

	p_line_RGB = p_RGB;
	p_line_Y = p_Y;
	p_line_UV = p_UV;


	for(j = 0; j< half_height; j++){

		p_mov_RGB = p_line_RGB;
		p_mov_Y = p_line_Y;

		p_mov_UV = p_line_UV;

		for(i = 0; i< half_width; i++){

			unsigned char Y;
			Y = p_mov_Y[0];

			p_mov_UV[0] = BY2U( p_mov_RGB[2], Y);
			p_mov_UV[1] = RY2V( p_mov_RGB[0], Y);

			p_mov_RGB += 6;
			p_mov_Y += 2;

			p_mov_UV += 2;
		}/*for i */

		p_line_RGB += double_width_RGB;
		p_line_Y += line_step_Y;

		p_line_UV += line_Step_UV;
	}/*for j*/

	return 0;
}

/**********************************************************************/

int RGBtoYV12(unsigned char *p_RGB, int width, int height,
			  unsigned char *p_Y, unsigned char *p_U, unsigned char *p_V)
{
	RGBtoY(p_RGB, width, height, width, p_Y);

	if(NULL != p_U)
		RGB32toV420(p_RGB + 0, width, height, p_Y, width, p_V);

	if(NULL != p_V)
		RGBtoU420(p_RGB + 2, width, height, p_Y, width, p_U);
	return 0;
}

/**********************************************************************/

int RGBtoNV12(unsigned char *p_RGB, int width, int height,
			  unsigned char *p_Y, unsigned char *p_UV)
{
	RGBtoY(p_RGB, width, height, width, p_Y);

	RGBtoUV420Interleave(p_RGB, width, height, p_Y, width, p_UV);

	return 0;
}

/**********************************************************************/
