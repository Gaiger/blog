#include <QDebug>

#include "H264MediaSubsession.h"
#include "CameraRTSPServer.h"

CameraRTSPServer::CameraRTSPServer(QObject *p_grabber,
								   unsigned short port)
	:
	m_p_h264_nal_factory(nullptr),
	m_port(port),
	m_p_scheduler(nullptr),
	m_p_rtsp_server(nullptr),
	m_p_server_media_session(nullptr),
	m_is_to_stop_scheduler(false)

{
	m_p_h264_nal_factory = new H264NalFactory(p_grabber);
	m_port = port;
}

/**********************************************************************/

CameraRTSPServer::~CameraRTSPServer(void)
{
	Stop();

	m_p_h264_nal_factory->SetEnabled(false);

	if(nullptr != m_p_h264_nal_factory)
		delete m_p_h264_nal_factory;
	m_p_h264_nal_factory = nullptr;
}

/**********************************************************************/

void CameraRTSPServer::Stop(void)
{
	qDebug() << Q_FUNC_INFO;

	if(false == QThread::isRunning())
			return ;

	m_is_to_stop_scheduler = 1;

	while(true == QThread::isRunning())
		QThread::wait(50);
}

/**********************************************************************/

void CameraRTSPServer::run(void)
{
	if(0 > CreateLive555Objects())
		return ;

	AnnounceStream(m_p_rtsp_server, m_p_server_media_session);

	m_is_to_stop_scheduler = 0;
	m_p_scheduler->doEventLoop(&m_is_to_stop_scheduler);

	DestroyLive555Objects();
}

/**********************************************************************/

int CameraRTSPServer::CreateLive555Objects(void)
{
	m_p_scheduler = BasicTaskScheduler::createNew();
	m_p_env = BasicUsageEnvironment::createNew(*m_p_scheduler);

#define MAX_PACKET_BUFFER_SIZE				(256 * 1024)
	/*IMPORTANT :: increase live 555 internal buffer size*/
	OutPacketBuffer::maxSize = MAX_PACKET_BUFFER_SIZE;

	UserAuthenticationDatabase* p_auth_database;
	p_auth_database= nullptr;


	m_p_rtsp_server
			= RTSPServer::createNew(*m_p_env, m_port, p_auth_database);

	if( nullptr == m_p_rtsp_server)
	{
		*m_p_env << "Failed to create RTSP server: "
				 << m_p_env->getResultMsg() << "\n";
		return -1;
	}

	{
		char description_string[64];

		memset(&description_string[0], 0, sizeof(description_string));

		snprintf(&description_string[0], sizeof(description_string),
				"Session streamed by \"CameraRTSPServer-%d\"", m_port);

		char stream_name[16];

		memset(&stream_name[0], 0, sizeof(stream_name));

		snprintf(&stream_name[0], sizeof(stream_name), "h264");

		m_p_server_media_session
				= ServerMediaSession::createNew(*m_p_env, &stream_name[0], nullptr,
												&description_string[0]);

		boolean is_reuse_first_source;
		is_reuse_first_source = true;


		OnDemandServerMediaSubsession *p_sms;

		p_sms = new H264MediaSubsession(*m_p_env,
										is_reuse_first_source,
										m_p_h264_nal_factory);

		m_p_server_media_session->addSubsession(p_sms);

		m_p_rtsp_server->addServerMediaSession(m_p_server_media_session);
	}
	return 0;
}

/**********************************************************************/

void CameraRTSPServer::DestroyLive555Objects(void)
{
	if(nullptr != m_p_server_media_session)
	{
		if(nullptr != m_p_rtsp_server)
			m_p_rtsp_server->removeServerMediaSession(
						m_p_server_media_session);
	}
	m_p_server_media_session = nullptr;

	if(nullptr != m_p_rtsp_server)
		Medium::close(m_p_rtsp_server);
	m_p_rtsp_server = nullptr;

	m_p_env->reclaim();

	if(nullptr != m_p_scheduler)
		delete m_p_scheduler;
	m_p_scheduler = nullptr;
}

/**********************************************************************/

void CameraRTSPServer::AnnounceStream(RTSPServer* m_p_rtsp_server,
									  ServerMediaSession *m_p_media_session)
{
	char* p_url;
	p_url = m_p_rtsp_server->rtspURL(m_p_media_session);

	UsageEnvironment &ref_env= m_p_rtsp_server->envir();

	ref_env << "\n\"" << m_p_media_session->streamName() <<
			   "\" stream \r\n";

	ref_env << "Play this stream using the URL \"" <<
			   p_url << "\"\r\n";


	delete[] p_url;
}

/**********************************************************************/
