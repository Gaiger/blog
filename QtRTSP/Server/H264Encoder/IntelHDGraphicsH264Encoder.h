#ifndef INTELHDGRAPHICSH264ENCODER_H
#define INTELHDGRAPHICSH264ENCODER_H

#include <QSize>
#include <QMutex>

#include "H264Encoder.h"

#include "mfxdefs.h"
#include "mfxcommon.h"
#include "mfxmvc.h"
#include "mfxvideo.h"
#include "mfxvideo++.h"
#include "mfxstructures.h"
#include "mfxsession.h"


class IntelHDGraphicsH264Encoder : public H264Encoder
{
public:
	static bool IsHardwareSupport(QSize resolution);

public:
	IntelHDGraphicsH264Encoder(void);
	~IntelHDGraphicsH264Encoder(void) Q_DECL_OVERRIDE;

public :
	int Init(int width, int height) Q_DECL_OVERRIDE;
	void Close(void) Q_DECL_OVERRIDE;

	QQueue<int> Encode(unsigned char *p_frame_data,
					   QQueue<QByteArray> &ref_queue) Q_DECL_OVERRIDE;
private:
	mfxStatus ExtendMfxBitstream(mfxU32 new_size);

private:
	int m_width, m_height;
	MFXVideoSession m_session;
	MFXVideoENCODE *m_p_enc;
	mfxU8 *m_p_surface_buffers;
	mfxFrameSurface1 **m_pp_encoding_surfaces;
	mfxU16 m_num_surfaces;
	mfxBitstream m_mfx_bitsream;

#if !defined(_NO_SPSPPS)
	mfxExtCodingOptionSPSPPS m_spspps_coding_option;
	mfxU8 m_sps_buffer[128];
	mfxU8 m_pps_buffer[128];
#endif

private :
	QMutex m_mutex;
};

#endif // INTELHDGRAPHICSH264ENCODER_H
