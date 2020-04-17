#ifndef X264ENCODER_H
#define X264ENCODER_H

#include <QObject>
#include <QMutex>

#include "H264Encoder.h"

#define X264_API_IMPORTS
extern "C"
{
#include "x264.h"
}


class X264Encoder : public H264Encoder
{
public:
	X264Encoder(void);
	~X264Encoder(void) Q_DECL_OVERRIDE;

public :
	int Init(int width, int height) Q_DECL_OVERRIDE;
	void Close(void) Q_DECL_OVERRIDE;

	QQueue<int> Encode(unsigned char *p_frame_data,
					   QQueue<QByteArray> &ref_queue) Q_DECL_OVERRIDE;

private:
	void SetParameters(x264_param_t &ref_param, int bitrate_in_kbps);
private:
	QMutex m_mutex;

	x264_t *m_x264_handle;
	x264_picture_t m_pic_in;

	int m_width, m_height;
};

#endif // X264ENCODER_H
