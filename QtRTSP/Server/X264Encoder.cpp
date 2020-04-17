
#include "rgb2yuv.h"
#include "X264Encoder.h"


X264Encoder::X264Encoder(void):
	m_x264_handle(nullptr),
	m_width(0), m_height(0)
{
	memset(&m_pic_in, 0, sizeof(x264_picture_t));
	init_lookup_table();
}

/**********************************************************************/

X264Encoder::~X264Encoder(void)
{
	Close();
}

/**********************************************************************/

int X264Encoder::Init(int width, int height)
{
	Close();
	QMutexLocker locker(&m_mutex);

	m_width = width;
	m_height = height;

	x264_param_t param;

	SetParameters(param, 2 * 1024);

	m_x264_handle = x264_encoder_open(&param);

	x264_picture_alloc(&m_pic_in, X264_CSP_I420 , width, height);

	return 0;
}

/**********************************************************************/

void X264Encoder::Close(void)
{
	QMutexLocker locker(&m_mutex);

	{
		x264_picture_t dummy_pic;
		memset(&dummy_pic, 0, sizeof(x264_picture_t));
		if(0 == memcmp(&dummy_pic, &m_pic_in, sizeof(x264_picture_t)))
			x264_picture_clean(&m_pic_in);

		memset(&m_pic_in, 0, sizeof(x264_picture_t));
	}


	if(nullptr != m_x264_handle)
		x264_encoder_close(m_x264_handle);
	m_x264_handle = nullptr;
}

/**********************************************************************/

void X264Encoder::SetParameters(x264_param_t &ref_param, int bitrate_in_kbps)
{
	x264_param_default_preset(&ref_param, "ultrafast", "zerolatency");

	ref_param.i_threads = 1;
	ref_param.i_bframe = 0;

	ref_param.b_repeat_headers = 1;

	ref_param.i_width = m_width;
	ref_param.i_height = m_height;

	ref_param.i_keyint_max = 60;
	ref_param.i_keyint_min = 15;
	ref_param.b_intra_refresh = 1;

	ref_param.rc.i_bitrate = bitrate_in_kbps;
	ref_param.rc.i_qp_max = 35;
	ref_param.rc.i_qp_min = 5;

	ref_param.b_annexb = 1;
}


/**********************************************************************/
//#include <QFile>

QQueue<int> X264Encoder::Encode(unsigned char *p_frame_data, QQueue<QByteArray> &ref_queue)
{
	QMutexLocker locker(&m_mutex);

	QQueue<int> h264_nal_size_queue;

	if(nullptr == m_x264_handle)
		return h264_nal_size_queue;

	rgb24_to_yuv420(m_width, m_height, p_frame_data, m_pic_in.img.plane[0], 0);
	x264_nal_t *p_nal;
	int i_nal;

	x264_picture_t pic_out;

	int encoded_size;
	encoded_size = x264_encoder_encode(m_x264_handle, &p_nal, &i_nal, &m_pic_in, &pic_out);

#if(0)
	QFile out_file;
	out_file.setFileName("AA.h264");
	out_file.open(QIODevice::WriteOnly| QIODevice::Append);
#endif
	for(int i = 0; i < i_nal; i++){

		QByteArray one_nal;
		one_nal = QByteArray((const char*)p_nal[i].p_payload, p_nal[i].i_payload);
#if(0)
		out_file.write(one_nal);
#endif
		ref_queue.enqueue(one_nal);
		h264_nal_size_queue.enqueue(p_nal[i].i_payload);
	}/*for i*/

#if(0)
	out_file.flush();
#endif
	return h264_nal_size_queue;
}
