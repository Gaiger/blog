#ifdef _WIN32
#include <windows.h>
#endif

#include <QApplication>

#include "ImageWidget.h"
#include "CameraFrameGrabber.h"

int main(int argc, char *argv[])
{
#ifdef _WIN32
	if (AttachConsole(ATTACH_PARENT_PROCESS)) {
		freopen("CONOUT$", "w", stdout);
		freopen("CONOUT$", "w", stderr);
	}
#endif

	QApplication a(argc, argv);

	CameraFrameGrabber frame_grabber;
	ImageWidget w;

	QObject::connect(&frame_grabber, SIGNAL(ResolutionChanged(QSize)),
					 &w, SLOT(ChangeResolution(QSize)));

	QObject::connect(&frame_grabber, SIGNAL(FrameUpdated(QImage)),
					 &w, SLOT(UpdateFrame(QImage)));

	frame_grabber.Start();
	w.show();


	return a.exec();
}
