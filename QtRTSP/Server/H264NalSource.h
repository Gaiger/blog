#ifndef H264NALSOURCE_H
#define H264NALSOURCE_H

#include "FramedSource.hh"
#include "H264NalFactory.h"

class H264NalSource : public FramedSource
{
public:
	static H264NalSource *createNew(UsageEnvironment &ref_env,
									unsigned client_session_id,
									void *p_h264_nal_factory);

protected:
	H264NalSource(UsageEnvironment &ref_env,
				  H264NalFactory *m_p_h264_nal_factory);
	~H264NalSource(void);

public:
	virtual void doGetNextFrame();
	virtual unsigned int maxFrameSize() const;

public:
	virtual Boolean isH264VideoStreamFramer(void) const {
		return true; // default implementation
	}


protected:
	void DeliverFrame(void);

private:
	H264NalFactory *m_p_h264_nal_factory;
	unsigned char *m_p_frame_buffer;
};

#endif // H264NALSOURCE_H
