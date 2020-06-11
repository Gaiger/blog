#ifndef _RGB2YUV_H_
#define _RGB2YUV_H_

#ifdef __cplusplus
extern "C" {
#endif

void init_lookup_table();

// ת�������лᶪʧ������Ϣ
int rgb24_to_yuv420(int x_dim, int y_dim, unsigned char *bmp, unsigned char *yuv, int flip);

int rgb2yuv();

#ifdef __cplusplus
}
#endif

#endif  /* _RGB2YUV_H_ */