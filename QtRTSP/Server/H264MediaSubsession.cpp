

#include "H264MediaSubsession.h"
#include "H264VideoStreamDiscreteFramer.hh"
#include "H264VideoRTPSink.hh"

#include "CameraH264Source.h"


H264MediaSubsession::H264MediaSubsession(UsageEnvironment& ref_env,
										 bool is_reuse_first_source, void *p_grabber)
	: OnDemandServerMediaSubsession(ref_env, is_reuse_first_source),
	m_p_grabber(p_grabber),
	m_p_AuxSDPLine(nullptr), m_done_flag(0), m_p_dummy_Sink(nullptr)
{

}

/**********************************************************************/

H264MediaSubsession::~H264MediaSubsession(void)
{
	printf("__FUNCTION__ = %s\r\n", __FUNCTION__);
	delete[] m_p_AuxSDPLine;
}

/**********************************************************************/

static void afterPlayingDummy(void* p_client_data)
{
	H264MediaSubsession *session = (H264MediaSubsession*)p_client_data;
	session->afterPlayingDummy1();
}

/**********************************************************************/

void H264MediaSubsession::afterPlayingDummy1()
{
	envir().taskScheduler().unscheduleDelayedTask(nextTask());
	setDoneFlag();
}

/**********************************************************************/

static void checkForAuxSDPLine(void* p_client_data)
{
	H264MediaSubsession* session = (H264MediaSubsession*)p_client_data;
	session->checkForAuxSDPLine1();
}

/**********************************************************************/

void H264MediaSubsession::checkForAuxSDPLine1(void)
{
	char const* dasl;
	if(nullptr != m_p_AuxSDPLine)
	{
		setDoneFlag();
	}
	else if(m_p_dummy_Sink != nullptr && (dasl = m_p_dummy_Sink->auxSDPLine()) != nullptr)
	{
		m_p_AuxSDPLine = strDup(dasl);
		m_p_dummy_Sink = nullptr;
		setDoneFlag();
	}
	else
	{

		int uSecsDelay = 200 * 1000;
		nextTask() = envir().taskScheduler().scheduleDelayedTask(uSecsDelay, (TaskFunc*)checkForAuxSDPLine, this);
	}
}

/**********************************************************************/

char const* H264MediaSubsession::getAuxSDPLine(RTPSink* rtp_sink, FramedSource *p_input_source)
{
	if(m_p_AuxSDPLine != nullptr)
		return m_p_AuxSDPLine;


	if(nullptr == m_p_dummy_Sink)
	{
		m_p_dummy_Sink = rtp_sink;
		m_p_dummy_Sink->startPlaying(*p_input_source, afterPlayingDummy, this);
		checkForAuxSDPLine(this);
	}

	envir().taskScheduler().doEventLoop(&m_done_flag);
	return m_p_AuxSDPLine;
}

/**********************************************************************/

FramedSource* H264MediaSubsession::createNewStreamSource(unsigned client_session_id,
														 unsigned &ref_estimate_bitrate)
{
	ref_estimate_bitrate = 2 * 1024 * 1024;

	CameraH264Source *p_source;
	p_source = CameraH264Source::createNew(envir(),
		client_session_id, m_p_grabber);

	return H264VideoStreamDiscreteFramer::createNew(envir(), p_source);
}

/**********************************************************************/

RTPSink* H264MediaSubsession::createNewRTPSink(Groupsock* rtpGroupsock,
											   unsigned char rtpPayloadTypeIfDynamic,
											   FramedSource* inputSource)
{
	(void)inputSource;
	return H264VideoRTPSink::createNew(envir(), rtpGroupsock, rtpPayloadTypeIfDynamic);
}
