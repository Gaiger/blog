
#include <QtGlobal>
#ifdef Q_OS_WIN
#include <windows.h>
#endif

#include <QApplication>

#include "ImageWidget.h"
#include "CameraFrameGrabber.h"
#include "H264NalFactory.h"

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
	ImageWidget w;

	QObject::connect(&frame_grabber, SIGNAL(ResolutionChanged(QSize)),
					 &w, SLOT(ChangeResolution(QSize)));

	QObject::connect(&frame_grabber, SIGNAL(FrameUpdated(QImage)),
					 &w, SLOT(UpdateFrame(QImage)));

	H264NalFactory nal_factory;

	QObject::connect(&frame_grabber, SIGNAL(ResolutionChanged(QSize)),
					 &nal_factory, SLOT(SetResolution(QSize)));
	QObject::connect(&frame_grabber, SIGNAL(FrameUpdated(QImage)),
						 &nal_factory, SLOT(Encode(QImage)));

	frame_grabber.Start();
	nal_factory.SetEnabled(true);
	w.show();


	return a.exec();
}
