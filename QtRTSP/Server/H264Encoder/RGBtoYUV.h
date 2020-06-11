
#ifndef _RGBTOYV12_H_
#define _RGBTOYV12_H_

#ifdef __cplusplus
extern "C" {
#endif

int RGBtoYV12(unsigned char *p_RGB, int width, int height,
			  unsigned char *p_Y, unsigned char *p_U, unsigned char *p_V);

int RGBtoNV12(unsigned char *p_RGB, int width, int height,
			  unsigned char *p_Y, unsigned char *p_UV);

#ifdef __cplusplus
}
#endif

#endif
