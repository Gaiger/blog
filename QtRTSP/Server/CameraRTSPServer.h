#ifndef CAMERARTSPSERVER_H
#define CAMERARTSPSERVER_H

#include <QThread>

#include "liveMedia.hh"
#include "GroupsockHelper.hh"

#include "BasicUsageEnvironment.hh"

#include "H264NalFactory.h"

class CameraRTSPServer : public QThread
{
	Q_OBJECT
public:
	explicit CameraRTSPServer(QObject *p_grabber,
							  unsigned short port = 8554);

	~CameraRTSPServer(void) Q_DECL_OVERRIDE;

	void Stop(void);

protected:
	virtual void run(void) Q_DECL_OVERRIDE;
private:
	int CreateLive555Objects(void);
	void DestroyLive555Objects(void);
	void AnnounceStream(RTSPServer* p_rtsp_server,
						ServerMediaSession *m_p_media_session);

private:
	H264NalFactory *m_p_h264_nal_factory;

	unsigned short m_port;
	UsageEnvironment *m_p_env;

	TaskScheduler *m_p_scheduler;
	RTSPServer *m_p_rtsp_server;
	ServerMediaSession *m_p_server_media_session;
	volatile char m_is_to_stop_scheduler;


};

#endif // CAMERARTSPSERVER_H
