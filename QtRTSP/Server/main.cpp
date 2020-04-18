
#include <QtGlobal>


#include <QApplication>


#include "CameraFrameGrabber.h"

#include "ImageWidget.h"
#include "CameraRTSPServer.h"

#ifdef Q_OS_WIN
#include <windows.h>
#endif

int main(int argc, char *argv[])
{
#ifdef Q_OS_WIN
	if (AttachConsole(ATTACH_PARENT_PROCESS)) {
		freopen("CONOUT$", "w", stdout);
		freopen("CONOUT$", "w", stderr);
	}
#endif

	QApplication a(argc, argv);

	CameraFrameGrabber frame_grabber;

	ImageWidget w(&frame_grabber);
	CameraRTSPServer rtsp_server(&frame_grabber, 8554);


	frame_grabber.Start();
	rtsp_server.start();
	w.show();

	return a.exec();
}
