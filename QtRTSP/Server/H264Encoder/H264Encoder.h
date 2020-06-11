#ifndef H264ENCODER_H
#define H264ENCODER_H

#include <QQueue>

class H264Encoder
{
public:
	H264Encoder(void){}
	virtual ~H264Encoder(void){}

public :
	virtual int Init(int width, int height) = 0;
	virtual void Close(void) = 0;

	virtual QQueue<int> Encode(unsigned char *p_frame_data,
							   QQueue<QByteArray> &ref_queue) = 0;

};

#endif // H264ENCODER_H
