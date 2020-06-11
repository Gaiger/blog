#include <GroupsockHelper.hh> // for "gettimeofday()"
#include "CameraH264Source.h"


#define ONE_FRAME_BUFFER_SIZE			(256 * 1024)

CameraH264Source *CameraH264Source::createNew(UsageEnvironment& ref_env,
										unsigned client_session_id,
										void *p_nal_factory)
{
	(void)client_session_id;

	return new CameraH264Source(ref_env, (H264NalFactory*)p_nal_factory);
}

/**********************************************************************/

CameraH264Source::CameraH264Source(UsageEnvironment &ref_env,
							 H264NalFactory *p_nal_factory)
	:FramedSource(ref_env),
	m_p_h264_nal_factory(p_nal_factory), m_p_frame_buffer(nullptr)
{
	printf("%s\r\n", __FUNCTION__);

	m_p_frame_buffer = new unsigned char[ONE_FRAME_BUFFER_SIZE];
	memset(m_p_frame_buffer, 0, sizeof(unsigned char)*ONE_FRAME_BUFFER_SIZE);

	m_p_h264_nal_factory->SetEnabled(true);
}

/**********************************************************************/

CameraH264Source::~CameraH264Source(void)
{
	printf("\n%s\r\n", __FUNCTION__);

	m_p_h264_nal_factory->SetEnabled(false);
	delete[] m_p_frame_buffer; m_p_frame_buffer = nullptr;
}

/**********************************************************************/

void CameraH264Source::doGetNextFrame()
{
	DeliverFrame();
}

/**********************************************************************/

unsigned int CameraH264Source::maxFrameSize(void) const
{
	return ONE_FRAME_BUFFER_SIZE;
}

/**********************************************************************/

void CameraH264Source::DeliverFrame(void)
{
	FramedSource::fFrameSize = 0;
	FramedSource::fNumTruncatedBytes = 0;

	unsigned int h264_data_length;

	h264_data_length = (unsigned int)m_p_h264_nal_factory->GetH264Nal(
				m_p_frame_buffer, ONE_FRAME_BUFFER_SIZE);

	if(0 >= h264_data_length)
		return ;

	//printf("h264_data_length = %d\r\n", h264_data_length);
	int trancate;
	trancate = 0;

	{
		char trancate_4[4] = {0x00, 0x00, 0x00, 0x01};
		char trancate_3[3] = {0x00, 0x00, 0x01};

		if(0 == memcmp(&m_p_frame_buffer[0], &trancate_4[0], sizeof(trancate_4)))
		{
			trancate = 4;
		}
		else if(0 == memcmp(&m_p_frame_buffer[0], &trancate_3[0], sizeof(trancate_3)))
		{
			trancate = 3;
		}
	}

	FramedSource::fFrameSize = h264_data_length;
	FramedSource::fNumTruncatedBytes = 0;

	if (h264_data_length > FramedSource::fMaxSize)
	{
		FramedSource::fFrameSize = FramedSource::fMaxSize;
		FramedSource::fNumTruncatedBytes =
				h264_data_length - FramedSource::fMaxSize;
	}/*if */

	gettimeofday(&(FramedSource::fPresentationTime), nullptr);

	memmove(FramedSource::fTo, m_p_frame_buffer + trancate,
		FramedSource::fFrameSize - trancate);

	FramedSource::afterGetting(this);
}

/**********************************************************************/

