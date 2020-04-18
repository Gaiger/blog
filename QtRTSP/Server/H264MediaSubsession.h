#ifndef H264MEDIASUBSESSION_H
#define H264MEDIASUBSESSION_H

#include "OnDemandServerMediaSubsession.hh"


class H264MediaSubsession : public OnDemandServerMediaSubsession
{
public:
	H264MediaSubsession(UsageEnvironment& ref_env, bool is_reuse_first_source, void *p_grabber);
	virtual ~H264MediaSubsession(void);
	void setDoneFlag() { m_done_flag = 1; }

protected:
	virtual char const* getAuxSDPLine(RTPSink* p_rtp_sink, FramedSource *p_input_source);
	virtual FramedSource* createNewStreamSource(unsigned client_session_id,
												unsigned &ref_estimate_bitrate);

	virtual RTPSink* createNewRTPSink(Groupsock* rtpGroupsock,
									  unsigned char rtpPayloadTypeIfDynamic,
									  FramedSource* inputSource);

public:
	void checkForAuxSDPLine1(void);
	void afterPlayingDummy1(void);

private:
	void *m_p_grabber;
	char* m_p_AuxSDPLine;
	char m_done_flag;
	RTPSink *m_p_dummy_Sink;

};

#endif // H264MEDIASUBSESSION_H
