﻿#include "ImageWidget.h"

#include <QPainter>

ImageWidget::ImageWidget(QObject *p_frame_grabber,
						 QWidget *parent)
	: QWidget(parent)
{
	QObject::connect(p_frame_grabber, SIGNAL(ResolutionChanged(QSize)),
					 this, SLOT(ChangeResolution(QSize)));

	QObject::connect(p_frame_grabber, SIGNAL(FrameUpdated(QImage)),
					 this, SLOT(UpdateFrame(QImage)));

	QWidget::resize(320, 240);
	QWidget::setFixedSize(QWidget::size());
}

/**********************************************************************/

ImageWidget::~ImageWidget()
{

}

/**********************************************************************/

void ImageWidget::ChangeResolution(QSize resolution)
{
	QWidget::setFixedSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);

	QWidget::resize(resolution);
	QWidget::setFixedSize(QWidget::size());
}

/**********************************************************************/

void ImageWidget::UpdateFrame(const QImage image)
{
	m_image = image.copy();
	emit QWidget::update();
}

/**********************************************************************/

void ImageWidget::paintEvent(QPaintEvent *event)
{
	QPainter painter(this);

	if(false == m_image.isNull())
		painter.drawImage(QWidget::rect(), m_image, QWidget::rect());

	QWidget::paintEvent(event);
}
