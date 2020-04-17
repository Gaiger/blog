#ifndef H264NALFACTORY_H
#define H264NALFACTORY_H

#include <QObject>
#include <QSemaphore>
#include <QMutex>
#include <QSize>

#include <QImage>


#include "H264Encoder.h"


class H264NalFactory : public QObject
{
	Q_OBJECT
public:
	H264NalFactory(QObject *p_frame_grabber);
	~H264NalFactory(void) Q_DECL_OVERRIDE;

public:
	int GetH264Nal(unsigned char *p_data, int buffer_size);

public slots:
	void SetResolution(QSize resolution);
	int Encode(QImage image);

public :
	void SetEnabled(bool is_enabled);
	bool IsEnabled(void);
private:
	QSemaphore m_semaphore;
	QMutex m_mutex;
	QSize m_resolution;
	bool m_is_enabled;

	H264Encoder *m_p_h264_encoder;
	QQueue<QByteArray> m_nal_queue;
};

#endif // H264NALFACTORY_H
