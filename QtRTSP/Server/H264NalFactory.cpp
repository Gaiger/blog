
#include "X264Encoder.h"

#include "H264NalFactory.h"


H264NalFactory::H264NalFactory(QObject *p_frame_grabber)
	: QObject(nullptr),
	m_resolution(QSize(0, 0)),
	m_is_enabled(false),
	m_p_h264_encoder(nullptr)
{
	QObject::connect(p_frame_grabber, SIGNAL(ResolutionChanged(QSize)),
					 this, SLOT(SetResolution(QSize)));
	QObject::connect(p_frame_grabber, SIGNAL(FrameUpdated(QImage)),
						 this, SLOT(Encode(QImage)));
}

/**********************************************************************/

H264NalFactory::~H264NalFactory(void)
{
	printf("%s\r\n", __FUNCTION__);
	QMutexLocker lock(&m_mutex);

	m_semaphore.release();

	if(nullptr != m_p_h264_encoder)
	{
		m_p_h264_encoder->Close();
		delete m_p_h264_encoder;
	}

	m_p_h264_encoder = nullptr;

	m_nal_queue.clear();
}

/**********************************************************************/

void H264NalFactory::SetResolution(QSize resolution)
{
	if(nullptr != m_p_h264_encoder)
		delete m_p_h264_encoder;
	m_p_h264_encoder = nullptr;

	m_nal_queue.clear();

	m_p_h264_encoder = new X264Encoder();
	m_resolution = resolution;

	//m_p_h264_encoder->Init(resolution.width(), resolution.height());
}

/**********************************************************************/

int H264NalFactory::Encode(QImage image)
{
	QMutexLocker lock(&m_mutex);

	QQueue<int> nal_size_queue;
	int encoded_frames;
	encoded_frames = 0;

	if(true == image.isNull())
		goto Flag_Encoding_End;

	nal_size_queue = m_p_h264_encoder->Encode(image.bits(), m_nal_queue);
	m_semaphore.release(nal_size_queue.size());

	encoded_frames = 1;
Flag_Encoding_End:
	return encoded_frames;
}

/**********************************************************************/

int H264NalFactory::GetH264Nal(unsigned char *p_data, int buffer_size)
{
#define MAX_WAIT_TIME_IN_MS							(200)
	m_semaphore.tryAcquire(1, MAX_WAIT_TIME_IN_MS);

	if(0 == m_nal_queue.size())
			return 0;

	QByteArray nal;
	nal = m_nal_queue.dequeue();

	int len;
	len = nal.size();

	if(len > buffer_size)
		len = buffer_size;

	memcpy(p_data, nal.data(), len);
	return len;
}

/**********************************************************************/

void H264NalFactory::SetEnabled(bool is_enabled)
{
	if(false == m_nal_queue.isEmpty())
		m_nal_queue.clear();
	if(0 != m_semaphore.available())
		m_semaphore.acquire(m_semaphore.available());

	if(QSize(0, 0) == m_resolution)
	{
		m_is_enabled = false;
		return ;
	}

	m_is_enabled = is_enabled;

	if(false == is_enabled)
	{
		m_p_h264_encoder->Close();
	}
	else
	{
		m_p_h264_encoder->Init(m_resolution.width(),
							   m_resolution.height());
	}
}

/**********************************************************************/

bool H264NalFactory::IsEnabled(void)
{
	return m_is_enabled;
}

/**********************************************************************/
