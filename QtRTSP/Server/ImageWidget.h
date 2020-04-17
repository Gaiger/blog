#ifndef IMAGEWIDGET_H
#define IMAGEWIDGET_H

#include <QWidget>

class ImageWidget : public QWidget
{
	Q_OBJECT

public:
	ImageWidget(QObject *p_frame_grabber, QWidget *parent = nullptr);
	~ImageWidget(void) Q_DECL_OVERRIDE;

public slots:
	void ChangeResolution(QSize resolution);
	void UpdateFrame(const QImage image);

protected:
	virtual void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE;

private:
	QImage m_image;
};

#endif // IMAGEWIDGET_H
