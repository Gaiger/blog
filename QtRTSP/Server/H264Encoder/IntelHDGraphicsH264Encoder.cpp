/*
 * https://software.intel.com/en-us/media-sdk/training
 * http://www.bvbcode.com/code/elip6ug3-2913641
 * https://www.cnblogs.com/programmer-wfq/p/7147042.html
*/

#include <QThread>


#include "RGBtoYUV.h"
#include "IntelHDGraphicsH264Encoder.h"


bool IntelHDGraphicsH264Encoder::IsHardwareSupport(QSize resolution)
{
	MFXVideoSession session;

	mfxStatus sts;
	mfxIMPL impl;
	mfxVersion version;

	impl = MFX_IMPL_HARDWARE_ANY;
	version.Major = 1;
	version.Minor = 0;

	sts = session.Init(impl, &version);
	if (MFX_ERR_NONE != sts)
	{
		printf("MFXVideoSession.Init  MFX_IMPL_HARDWARE_ANY failed, %d\r\n", sts);
		return false;
	}
	session.Close();

	impl = MFX_IMPL_HARDWARE;
	sts = session.Init( impl, &version);
	if (MFX_ERR_NONE != sts)
	{
		printf("MFXVideoSession.Init  MFX_IMPL_HARDWARE failed, %d\r\n", sts);
		return false;
	}


	mfxVideoParam params_in;
	memset(&params_in, 0, sizeof(mfxVideoParam));

	params_in.mfx.CodecId = MFX_CODEC_AVC;
#if(0)
	params_in.mfx.TargetUsage = MFX_TARGETUSAGE_BALANCED;
	params_in.mfx.TargetKbps = 2 * 1024;
	params_in.mfx.RateControlMethod = MFX_RATECONTROL_VBR;
	params_in.mfx.FrameInfo.FrameRateExtN = 30;
	params_in.mfx.FrameInfo.FrameRateExtD = 1;
#endif
	params_in.mfx.FrameInfo.FourCC = MFX_FOURCC_NV12;
	params_in.mfx.FrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
	params_in.mfx.FrameInfo.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
	params_in.mfx.FrameInfo.CropX = 0;
	params_in.mfx.FrameInfo.CropY = 0;
	params_in.mfx.FrameInfo.CropW = resolution.width();
	params_in.mfx.FrameInfo.CropH = resolution.height();
	params_in.mfx.FrameInfo.Width = resolution.width();
	params_in.mfx.FrameInfo.Height = resolution.height();
	params_in.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY;

	sts = MFXQueryIMPL(session, &impl);
	if (MFX_ERR_NONE != sts)
	{
		printf("MFXQueryIMPL failed, %d\r\n", sts);
		return false;
	}

	sts = MFXQueryVersion((mfxSession)session, &version);
	if (MFX_ERR_NONE != sts)
	{
		printf("MFXQueryVersion failed, %d\r\n", sts);
		return false;
	}

	MFXVideoENCODE enc(session);

	sts = enc.Query(&params_in, &params_in);
	if (MFX_ERR_NONE != sts)
	{
		printf("MFXVideoENCODE enc failed, %d\r\n", sts);
		return false;
	}

	enc.Close();

	return true;
}

/**********************************************************************/

IntelHDGraphicsH264Encoder::IntelHDGraphicsH264Encoder()
	:m_width(0), m_height(0),
	 m_p_enc(nullptr),
	 m_p_surface_buffers(nullptr),
	 m_pp_encoding_surfaces(nullptr),
	 m_num_surfaces(0)
{
	memset(&m_mfx_bitsream, 0, sizeof(mfxBitstream));
	Close();
}

/**********************************************************************/

IntelHDGraphicsH264Encoder::~IntelHDGraphicsH264Encoder(void)
{
	Close();
}

/**********************************************************************/

int IntelHDGraphicsH264Encoder::Init(int width, int height)
{
	Close();
	QMutexLocker locker(&m_mutex);

	int bitrate_in_kbps;
	bitrate_in_kbps = 2 * 1024;

	m_width = width;
	m_height = height;

	mfxStatus sts;
	mfxIMPL impl;
	mfxVersion version;

	version.Major = 1; version.Minor = 0;
	impl = MFX_IMPL_HARDWARE;



	sts = m_session.Init( impl, &version);
	if (MFX_ERR_NONE != sts)
	{
		printf("MFXVideoSession.Init  MFX_IMPL_HARDWARE failed, %d\r\n", sts);
		return -1;
	}

	mfxVideoParam params_in;
	memset(&params_in, 0, sizeof(mfxVideoParam));

	//low letency
	params_in.AsyncDepth = 1;
	params_in.mfx.NumRefFrame = 1;
	params_in.mfx.IdrInterval = 1;


	params_in.mfx.CodecId = MFX_CODEC_AVC;
	params_in.mfx.TargetUsage = MFX_TARGETUSAGE_1; // MFX_TARGETUSAGE_BEST_SPEED; MFX_TARGETUSAGE_BALANCED
	params_in.mfx.TargetKbps = bitrate_in_kbps* 3 / 4 ;
	params_in.mfx.MaxKbps = bitrate_in_kbps;
	params_in.mfx.RateControlMethod = MFX_RATECONTROL_VBR;
	params_in.mfx.FrameInfo.FrameRateExtN = 30;
	params_in.mfx.FrameInfo.FrameRateExtD = 1;

	params_in.mfx.GopRefDist = 1;
	params_in.mfx.GopPicSize = 25;


	params_in.mfx.TargetUsage = MFX_TARGETUSAGE_2;//MFX_TARGETUSAGE_BEST_QUALITY; //MFX_TARGETUSAGE_5;// MFX_TARGETUSAGE_BEST_SPEED;
	params_in.mfx.FrameInfo.FourCC = MFX_FOURCC_NV12;
	params_in.mfx.FrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
	params_in.mfx.FrameInfo.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
	params_in.mfx.FrameInfo.CropX = 0;
	params_in.mfx.FrameInfo.CropY = 0;
	params_in.mfx.FrameInfo.CropW = width;
	params_in.mfx.FrameInfo.CropH = height;
	params_in.mfx.FrameInfo.Width = width;
	params_in.mfx.FrameInfo.Height = height;
	params_in.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY;



	m_p_enc = new MFXVideoENCODE(m_session);

	sts = m_p_enc->Query(&params_in, &params_in);
	if (MFX_ERR_NONE != sts)
	{
		printf("MFXVideoENCODE.Query failed, %d\r\n", sts);
		return -2;
	}


	mfxFrameAllocRequest request;
	memset(&request, 0, sizeof(request));

	sts = m_p_enc->QueryIOSurf(&params_in, &request);
	if (MFX_ERR_NONE != sts)
	{
		printf("MFXVideoENCODE.QueryIOSurf failed, %d\r\n", sts);
		return -3;
	}

	m_num_surfaces = request.NumFrameSuggested;

	mfxU8 bitsPerPixel = (mfxU8)12;		// NV12 format is a 12 bits per pixel format
	mfxU32 surfaceSize = request.Info.Width * request.Info.Height * bitsPerPixel / 8;
	m_p_surface_buffers = (mfxU8*)malloc(surfaceSize * m_num_surfaces);

	m_pp_encoding_surfaces =
			(mfxFrameSurface1**)malloc(m_num_surfaces * sizeof(mfxFrameSurface1*));

	memset(m_pp_encoding_surfaces, 0, m_num_surfaces * sizeof(mfxFrameSurface1*));

	for (int i = 0; i < m_num_surfaces; i++) {

		m_pp_encoding_surfaces[i] =  (mfxFrameSurface1*)malloc(sizeof(mfxFrameSurface1));
		memset(m_pp_encoding_surfaces[i], 0, sizeof(mfxFrameSurface1));

		memcpy(&(m_pp_encoding_surfaces[i]->Info), &(params_in.mfx.FrameInfo),
			   sizeof(mfxFrameInfo));

		m_pp_encoding_surfaces[i]->Data.Y
				= &m_p_surface_buffers[surfaceSize * i];
		m_pp_encoding_surfaces[i]->Data.U
				= m_pp_encoding_surfaces[i]->Data.Y + width * height;
		m_pp_encoding_surfaces[i]->Data.V
				= m_pp_encoding_surfaces[i]->Data.U + 1;

		m_pp_encoding_surfaces[i]->Data.Pitch = width;
	}


	sts = m_p_enc->Init(&params_in);
	if (MFX_ERR_NONE != sts)
	{
		printf("MFXVideoENCODE.Init failed, %d\r\n", sts);
		return -4;
	}

#if !defined(_NO_SPSPPS)
	memset(&m_spspps_coding_option, 0, sizeof(mfxExtCodingOptionSPSPPS));

	m_spspps_coding_option.Header.BufferId = MFX_EXTBUFF_CODING_OPTION_SPSPPS;
	m_spspps_coding_option.Header.BufferSz = sizeof(mfxExtCodingOptionSPSPPS);

	m_spspps_coding_option.SPSBuffer = &m_sps_buffer[0];
	m_spspps_coding_option.SPSBufSize = sizeof(m_sps_buffer);
	m_spspps_coding_option.PPSBuffer = &m_pps_buffer[0];
	m_spspps_coding_option.PPSBufSize = sizeof(m_pps_buffer);

	mfxExtBuffer* extendedBuffers[1];
	extendedBuffers[0] = (mfxExtBuffer*) & m_spspps_coding_option;
#endif

	mfxVideoParam video_param;
	memset(&video_param, 0, sizeof(video_param));

#if !defined(_NO_SPSPPS)
	video_param.ExtParam = &extendedBuffers[0];
	video_param.NumExtParam = 1;
#endif

	sts = m_p_enc->GetVideoParam(&video_param);

	if (MFX_ERR_NONE != sts)
	{
		printf("MFXVideoENCODE.GetVideoParam failed, %d\r\n", sts);
		return -6;
	}

	//printf("SPS = %d, PPS = %d\r\n",
	//	   m_spspps_coding_option.SPSBufSize, m_spspps_coding_option.PPSBufSize);

	m_mfx_bitsream.MaxLength = video_param.mfx.BufferSizeInKB * 1024;
	m_mfx_bitsream.Data = (mfxU8*)malloc(m_mfx_bitsream.MaxLength);
	memset(m_mfx_bitsream.Data, 0, m_mfx_bitsream.MaxLength);

	return 0;
}

/**********************************************************************/

void IntelHDGraphicsH264Encoder::Close(void)
{
	QMutexLocker locker(&m_mutex);
	if(nullptr != m_p_enc)
	{
		m_p_enc->Close();
		delete m_p_enc;
	}
	m_p_enc = nullptr;

	if(nullptr != m_p_surface_buffers)
		free(m_p_surface_buffers);
	m_p_surface_buffers = nullptr;

	if(nullptr != m_pp_encoding_surfaces)
	{
		for(int i = 0; i < m_num_surfaces; i++){
			if(nullptr != m_pp_encoding_surfaces[i])
				free(m_pp_encoding_surfaces[i]);
			m_pp_encoding_surfaces[i] = nullptr;
		}
	}

	if(nullptr != m_mfx_bitsream.Data)
		free(m_mfx_bitsream.Data);
	m_mfx_bitsream.Data = nullptr;

	memset(&m_mfx_bitsream, 0, sizeof(mfxBitstream));
}

/**********************************************************************/

mfxStatus IntelHDGraphicsH264Encoder::ExtendMfxBitstream(mfxU32 new_size)
{
	mfxU8 *p_data;
	p_data = (mfxU8*)malloc(new_size);
	memset(p_data, 0, new_size);

	if(nullptr == p_data)
		return MFX_ERR_MEMORY_ALLOC;

	if(nullptr != m_mfx_bitsream.Data)
	{
		memmove(p_data, m_mfx_bitsream.Data + m_mfx_bitsream.DataOffset,
				m_mfx_bitsream.DataLength);
		free(m_mfx_bitsream.Data); m_mfx_bitsream.Data = nullptr;
	}
	m_mfx_bitsream.Data = p_data;
	m_mfx_bitsream.DataOffset = 0;
	m_mfx_bitsream.MaxLength = new_size;

	return MFX_ERR_NONE;
}

/**********************************************************************/

QQueue<int> IntelHDGraphicsH264Encoder::Encode(unsigned char *p_frame_data, QQueue<QByteArray> &ref_queue)
{
	QMutexLocker locker(&m_mutex);


	QQueue<int> h264_nal_size_queue;
	if(nullptr == m_p_enc)
		return h264_nal_size_queue;

	int encoding_surf_idx;
	encoding_surf_idx = 0;

	for(int i = 0; i < m_num_surfaces; i++){
		if(0 == m_pp_encoding_surfaces[i]->Data.Locked)
		{
			encoding_surf_idx = i;
			break;
		}
	}

	RGBtoNV12(p_frame_data, m_width, m_height,
			  m_pp_encoding_surfaces[encoding_surf_idx]->Data.Y,
			  m_pp_encoding_surfaces[encoding_surf_idx]->Data.U);


	mfxStatus sts;
	mfxSyncPoint encoding_sync_point;

	while(1)
	{
		sts = m_p_enc->EncodeFrameAsync(nullptr, m_pp_encoding_surfaces[encoding_surf_idx],
										&m_mfx_bitsream, &encoding_sync_point);

		if (MFX_ERR_NONE < sts && 0 == encoding_sync_point)
		{	// Repeat the call if warning and no output
			if (MFX_WRN_DEVICE_BUSY == sts)
				QThread::msleep(1);  // Wait if device is busy, then repeat the same call
		}
		else if (MFX_ERR_NONE < sts && encoding_sync_point)
		{
			sts = MFX_ERR_NONE;     // Ignore warnings if output is available
			break;
		}
		else if (MFX_ERR_NOT_ENOUGH_BUFFER == sts)
		{
			mfxVideoParam par;

			memset(&par, 0, sizeof(par));
			sts = m_p_enc->GetVideoParam(&par);

			//printf("require %d KB\r\n", par.mfx.BufferSizeInKB);
			ExtendMfxBitstream( par.mfx.BufferSizeInKB * 1024);

			break;
		} else
		{
			break;
		}

	}

	QByteArray array;

	if(MFX_ERR_NONE != sts)
	{
		printf("MFXVideoENCODE.EncodeFrameAsync fail, %d\r\n", sts);
		goto Flag_Encode_End;
	}

	sts = m_session.SyncOperation(encoding_sync_point, 60000);
	if(MFX_ERR_NONE != sts)
	{
		printf("MFXVideoSession.SyncOperation fail, %d\r\n", sts);
		goto Flag_Encode_End;
	}

	//printf("Encoding done :: %d \r\n",  m_mfx_bitsream.DataLength);

	array = QByteArray((char*)m_mfx_bitsream.Data + m_mfx_bitsream.DataOffset,
								m_mfx_bitsream.DataLength);

#if !defined(_NO_SPSPPS)
	if( MFX_FRAMETYPE_IDR & m_mfx_bitsream.FrameType)
	{
		//printf("MFX_FRAMETYPE_IDR\r\n");
		ref_queue.enqueue(QByteArray((char*)m_spspps_coding_option.SPSBuffer,
									 m_spspps_coding_option.SPSBufSize));
		h264_nal_size_queue.enqueue( m_spspps_coding_option.SPSBufSize);
		ref_queue.enqueue(QByteArray((char*)m_spspps_coding_option.PPSBuffer,
									 m_spspps_coding_option.PPSBufSize));
		h264_nal_size_queue.enqueue( m_spspps_coding_option.PPSBufSize);
	}
#endif

	ref_queue.enqueue(array);
	h264_nal_size_queue.enqueue( m_mfx_bitsream.DataLength);

	m_mfx_bitsream.DataLength = 0;

Flag_Encode_End:
	return h264_nal_size_queue;
}

/**********************************************************************/
