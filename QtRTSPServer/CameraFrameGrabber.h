#ifndef CAMERAFRAMEGRABBER_H
#define CAMERAFRAMEGRABBER_H

#include <QObject>

#include <QAbstractVideoSurface>
#include <QCamera>


class CameraFrameGrabber : public QAbstractVideoSurface
{
	Q_OBJECT
public:
	explicit CameraFrameGrabber(QAbstractVideoSurface *parent = nullptr);

	~CameraFrameGrabber(void) Q_DECL_OVERRIDE;

public:
	bool Start(void);
	void Stop(void);

public:
	signals:
	void ResolutionChanged(QSize resolution);
	void FrameUpdated(QImage image);

public:
	virtual bool present(const QVideoFrame &frame) Q_DECL_OVERRIDE;

	virtual QList<QVideoFrame::PixelFormat>
		supportedPixelFormats(QAbstractVideoBuffer::HandleType type) const Q_DECL_OVERRIDE;

private slots:
	void ChangeCameraSetting(QCamera::State state);

private:
	QCamera	*m_p_camera;
};

#endif // CAMERAFRAMEGRABBER_H
