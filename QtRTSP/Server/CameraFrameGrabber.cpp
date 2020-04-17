#include <QDebug>

#include <QCameraInfo>
#include "CameraFrameGrabber.h"


void YUYVtoRGB(int width, int height, unsigned char *yuyv_image, unsigned char *rgb_image)
{
	int y;
	int cr;
	int cb;

	double r;
	double g;
	double b;
	int i, j;

	for (i = 0, j = 0; i < width * height * 3; i+=6, j+=4) {
		//first pixel
		y = yuyv_image[j];
		cb = yuyv_image[j+1];
		cr = yuyv_image[j+3];

		r = y + (1.4065 * (cr - 128));
		g = y - (0.3455 * (cb - 128)) - (0.7169 * (cr - 128));
		b = y + (1.7790 * (cb - 128));

		//This prevents colour distortions in your rgb image
		if (r < 0) r = 0;
		else if (r > 255) r = 255;
		if (g < 0) g = 0;
		else if (g > 255) g = 255;
		if (b < 0) b = 0;
		else if (b > 255) b = 255;

		rgb_image[i] = (unsigned char)r;
		rgb_image[i+1] = (unsigned char)g;
		rgb_image[i+2] = (unsigned char)b;

		//second pixel
		y = yuyv_image[j+2];
		cb = yuyv_image[j+1];
		cr = yuyv_image[j+3];

		r = y + (1.4065 * (cr - 128));
		g = y - (0.3455 * (cb - 128)) - (0.7169 * (cr - 128));
		b = y + (1.7790 * (cb - 128));

		if (r < 0) r = 0;
		else if (r > 255) r = 255;
		if (g < 0) g = 0;
		else if (g > 255) g = 255;
		if (b < 0) b = 0;
		else if (b > 255) b = 255;

		rgb_image[i+3] = (unsigned char)r;
		rgb_image[i+4] = (unsigned char)g;
		rgb_image[i+5] = (unsigned char)b;
	}
}

/**********************************************************************/

CameraFrameGrabber::CameraFrameGrabber(QAbstractVideoSurface *parent)
	: QAbstractVideoSurface(parent),
	  m_p_camera(nullptr)
{

}

/**********************************************************************/

CameraFrameGrabber::~CameraFrameGrabber(void)
{
	Stop();
}

/**********************************************************************/

bool CameraFrameGrabber::present(const QVideoFrame &frame)
{
	//qDebug() << "grabbed pixelFormat is " << cloneFrame.pixelFormat();
	QVideoFrame cloneFrame(frame);

	cloneFrame.map(QAbstractVideoBuffer::ReadOnly);

	QImage grabbed_image;

	if(QVideoFrame::Format_YUYV == cloneFrame.pixelFormat())
	{
		QByteArray temp;
		temp.resize(cloneFrame.width() * cloneFrame.height() * 3);

		YUYVtoRGB(cloneFrame.width(), cloneFrame.height(),
				  cloneFrame.bits(), (unsigned char*)temp.data());

		grabbed_image =  QImage((unsigned char*)temp.data(),
								cloneFrame.width(),
								cloneFrame.height(),
								cloneFrame.width() * 3,
								QImage::Format_RGB888).copy();
	}
#if(0)
	else if(QVideoFrame::Format_RGB32 == cloneFrame.pixelFormat())
	{
		grabbed_image = QImage(cloneFrame.bits(), cloneFrame.width(),
							   cloneFrame.height(),
							   cloneFrame.width() * 4,
							   QImage::Format_RGB32).copy();
	}
#endif
	cloneFrame.unmap();

	emit FrameUpdated(grabbed_image);

	return true;
}

/**********************************************************************/

QList<QVideoFrame::PixelFormat>
	CameraFrameGrabber::supportedPixelFormats(
		QAbstractVideoBuffer::HandleType type) const
{
	if (type != QAbstractVideoBuffer::NoHandle)
		return QList<QVideoFrame::PixelFormat>();

	QList<QVideoFrame::PixelFormat> pixel_fmt;
#if(0)
	pixel_fmt.append(QVideoFrame::Format_RGB24);
	pixel_fmt.append(QVideoFrame::Format_RGB32);
#endif
	pixel_fmt.append(QVideoFrame::Format_YUYV);

	return pixel_fmt;
}

/**********************************************************************/

void CameraFrameGrabber::Stop(void)
{
	if(nullptr == m_p_camera)
			return;

	m_p_camera->stop();
	delete m_p_camera;
	m_p_camera = nullptr;

	QAbstractVideoSurface::stop();
}

/**********************************************************************/

bool CameraFrameGrabber::Start(void)
{
	if(nullptr != m_p_camera)
		return true;

	const QList<QCameraInfo> available_cameras_info_list
			= QCameraInfo::availableCameras();

	if(0 == available_cameras_info_list.size())
	{
		qDebug() << "no camera found";
		return false;
	}

	for(int i = 0; i < available_cameras_info_list.size(); i++){
			qDebug() << available_cameras_info_list.at(i).description();
		}

	m_p_camera = new QCamera(available_cameras_info_list.at(2));

	connect(m_p_camera, SIGNAL(stateChanged(QCamera::State)),
				this, SLOT(ChangeCameraSetting(QCamera::State)));

	m_p_camera->setViewfinder(this);
	m_p_camera->start();

	return true;
}

/**********************************************************************/


void CameraFrameGrabber::ChangeCameraSetting(QCamera::State state)
{
	if(QCamera::ActiveState != state)
		return ;

	QCameraViewfinderSettings viewfinderSettings;
	viewfinderSettings = m_p_camera->viewfinderSettings();

	//qDebug() << m_p_camera->supportedViewfinderResolutions();
	//qDebug() << m_p_camera->supportedViewfinderPixelFormats();
	//viewfinderSettings.setResolution(640, 480);

	//viewfinderSettings.setMaximumFrameRate(30.0);
	//viewfinderSettings.setMinimumFrameRate(30.0);

	m_p_camera->setViewfinderSettings(viewfinderSettings);
	viewfinderSettings = m_p_camera->viewfinderSettings();

	qDebug() << viewfinderSettings.resolution();
	qDebug() << viewfinderSettings.pixelFormat();
	qDebug() << viewfinderSettings.maximumFrameRate() << viewfinderSettings.minimumFrameRate();

	emit ResolutionChanged(viewfinderSettings.resolution());
}
