#ifndef CameraH264Source_H
#define CameraH264Source_H

#include "FramedSource.hh"
#include "H264NalFactory.h"

class CameraH264Source : public FramedSource
{
public:
	static CameraH264Source *createNew(UsageEnvironment &ref_env,
			unsigned client_session_id,
			void *p_h264_nal_factory);

protected:
	CameraH264Source(UsageEnvironment &ref_env,
			 H264NalFactory *m_p_h264_nal_factory);
	~CameraH264Source(void);

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

#endif // CameraH264Source_H
