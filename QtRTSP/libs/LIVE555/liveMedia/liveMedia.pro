QT -= gui core

TEMPLATE = lib
CONFIG += staticlib

CONFIG += c++11


DEFINES += NO_OPENSSL
# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


LIVE555_PATH = $$PWD/..

INCLUDEPATH += $$LIVE555_PATH/BasicUsageEnvironment/include
INCLUDEPATH += $$LIVE555_PATH/groupsock/include
INCLUDEPATH += $$LIVE555_PATH/liveMedia/include
INCLUDEPATH += $$LIVE555_PATH/UsageEnvironment/include

SOURCES += \
    AC3AudioFileServerMediaSubsession.cpp \
    AC3AudioRTPSink.cpp \
    AC3AudioRTPSource.cpp \
    AC3AudioStreamFramer.cpp \
    ADTSAudioFileServerMediaSubsession.cpp \
    ADTSAudioFileSource.cpp \
    AMRAudioFileServerMediaSubsession.cpp \
    AMRAudioFileSink.cpp \
    AMRAudioFileSource.cpp \
    AMRAudioRTPSink.cpp \
    AMRAudioRTPSource.cpp \
    AMRAudioSource.cpp \
    AVIFileSink.cpp \
    AudioInputDevice.cpp \
    AudioRTPSink.cpp \
    Base64.cpp \
    BasicUDPSink.cpp \
    BasicUDPSource.cpp \
    BitVector.cpp \
    ByteStreamFileSource.cpp \
    ByteStreamMemoryBufferSource.cpp \
    ByteStreamMultiFileSource.cpp \
    DVVideoFileServerMediaSubsession.cpp \
    DVVideoRTPSink.cpp \
    DVVideoRTPSource.cpp \
    DVVideoStreamFramer.cpp \
    DeviceSource.cpp \
    DigestAuthentication.cpp \
    EBMLNumber.cpp \
    FileServerMediaSubsession.cpp \
    FileSink.cpp \
    FramedFileSource.cpp \
    FramedFilter.cpp \
    FramedSource.cpp \
    GSMAudioRTPSink.cpp \
    GenericMediaServer.cpp \
    H261VideoRTPSource.cpp \
    H263plusVideoFileServerMediaSubsession.cpp \
    H263plusVideoRTPSink.cpp \
    H263plusVideoRTPSource.cpp \
    H263plusVideoStreamFramer.cpp \
    H263plusVideoStreamParser.cpp \
    H264VideoFileServerMediaSubsession.cpp \
    H264VideoFileSink.cpp \
    H264VideoRTPSink.cpp \
    H264VideoRTPSource.cpp \
    H264VideoStreamDiscreteFramer.cpp \
    H264VideoStreamFramer.cpp \
    H264or5VideoFileSink.cpp \
    H264or5VideoRTPSink.cpp \
    H264or5VideoStreamDiscreteFramer.cpp \
    H264or5VideoStreamFramer.cpp \
    H265VideoFileServerMediaSubsession.cpp \
    H265VideoFileSink.cpp \
    H265VideoRTPSink.cpp \
    H265VideoRTPSource.cpp \
    H265VideoStreamDiscreteFramer.cpp \
    H265VideoStreamFramer.cpp \
    HLSSegmenter.cpp \
    InputFile.cpp \
    JPEG2000VideoRTPSink.cpp \
    JPEG2000VideoRTPSource.cpp \
    JPEGVideoRTPSink.cpp \
    JPEGVideoRTPSource.cpp \
    JPEGVideoSource.cpp \
    Locale.cpp \
    MP3ADU.cpp \
    MP3ADURTPSink.cpp \
    MP3ADURTPSource.cpp \
    MP3ADUTranscoder.cpp \
    MP3ADUdescriptor.cpp \
    MP3ADUinterleaving.cpp \
    MP3AudioFileServerMediaSubsession.cpp \
    MP3AudioMatroskaFileServerMediaSubsession.cpp \
    MP3FileSource.cpp \
    MP3Internals.cpp \
    MP3InternalsHuffman.cpp \
    MP3InternalsHuffmanTable.cpp \
    MP3StreamState.cpp \
    MP3Transcoder.cpp \
    MPEG1or2AudioRTPSink.cpp \
    MPEG1or2AudioRTPSource.cpp \
    MPEG1or2AudioStreamFramer.cpp \
    MPEG1or2Demux.cpp \
    MPEG1or2DemuxedElementaryStream.cpp \
    MPEG1or2DemuxedServerMediaSubsession.cpp \
    MPEG1or2FileServerDemux.cpp \
    MPEG1or2VideoFileServerMediaSubsession.cpp \
    MPEG1or2VideoRTPSink.cpp \
    MPEG1or2VideoRTPSource.cpp \
    MPEG1or2VideoStreamDiscreteFramer.cpp \
    MPEG1or2VideoStreamFramer.cpp \
    MPEG2IndexFromTransportStream.cpp \
    MPEG2TransportFileServerMediaSubsession.cpp \
    MPEG2TransportStreamAccumulator.cpp \
    MPEG2TransportStreamDemux.cpp \
    MPEG2TransportStreamDemuxedTrack.cpp \
    MPEG2TransportStreamFramer.cpp \
    MPEG2TransportStreamFromESSource.cpp \
    MPEG2TransportStreamFromPESSource.cpp \
    MPEG2TransportStreamIndexFile.cpp \
    MPEG2TransportStreamMultiplexor.cpp \
    MPEG2TransportStreamParser.cpp \
    MPEG2TransportStreamParser_PAT.cpp \
    MPEG2TransportStreamParser_PMT.cpp \
    MPEG2TransportStreamParser_STREAM.cpp \
    MPEG2TransportStreamTrickModeFilter.cpp \
    MPEG2TransportUDPServerMediaSubsession.cpp \
    MPEG4ESVideoRTPSink.cpp \
    MPEG4ESVideoRTPSource.cpp \
    MPEG4GenericRTPSink.cpp \
    MPEG4GenericRTPSource.cpp \
    MPEG4LATMAudioRTPSink.cpp \
    MPEG4LATMAudioRTPSource.cpp \
    MPEG4VideoFileServerMediaSubsession.cpp \
    MPEG4VideoStreamDiscreteFramer.cpp \
    MPEG4VideoStreamFramer.cpp \
    MPEGVideoStreamFramer.cpp \
    MPEGVideoStreamParser.cpp \
    MatroskaDemuxedTrack.cpp \
    MatroskaFile.cpp \
    MatroskaFileParser.cpp \
    MatroskaFileServerDemux.cpp \
    MatroskaFileServerMediaSubsession.cpp \
    Media.cpp \
    MediaSession.cpp \
    MediaSink.cpp \
    MediaSource.cpp \
    MultiFramedRTPSink.cpp \
    MultiFramedRTPSource.cpp \
    OggDemuxedTrack.cpp \
    OggFile.cpp \
    OggFileParser.cpp \
    OggFileServerDemux.cpp \
    OggFileServerMediaSubsession.cpp \
    OggFileSink.cpp \
    OnDemandServerMediaSubsession.cpp \
    OutputFile.cpp \
    PassiveServerMediaSubsession.cpp \
    ProxyServerMediaSession.cpp \
    QCELPAudioRTPSource.cpp \
    QuickTimeFileSink.cpp \
    QuickTimeGenericRTPSource.cpp \
    RTCP.cpp \
    RTPInterface.cpp \
    RTPSink.cpp \
    RTPSource.cpp \
    RTSPClient.cpp \
    RTSPCommon.cpp \
    RTSPRegisterSender.cpp \
    RTSPServer.cpp \
    RTSPServerRegister.cpp \
    RawVideoRTPSink.cpp \
    RawVideoRTPSource.cpp \
    SIPClient.cpp \
    ServerMediaSession.cpp \
    SimpleRTPSink.cpp \
    SimpleRTPSource.cpp \
    StreamParser.cpp \
    StreamReplicator.cpp \
    T140TextRTPSink.cpp \
    TLSState.cpp \
    TextRTPSink.cpp \
    TheoraVideoRTPSink.cpp \
    TheoraVideoRTPSource.cpp \
    VP8VideoRTPSink.cpp \
    VP8VideoRTPSource.cpp \
    VP9VideoRTPSink.cpp \
    VP9VideoRTPSource.cpp \
    VideoRTPSink.cpp \
    VorbisAudioRTPSink.cpp \
    VorbisAudioRTPSource.cpp \
    WAVAudioFileServerMediaSubsession.cpp \
    WAVAudioFileSource.cpp \
    ourMD5.cpp \
    rtcp_from_spec.c \
    uLawAudioFilter.cpp


HEADERS += \
    EBMLNumber.hh \
    H263plusVideoStreamParser.hh \
    MP3ADUdescriptor.hh \
    MP3AudioMatroskaFileServerMediaSubsession.hh \
    MP3Internals.hh \
    MP3InternalsHuffman.hh \
    MP3StreamState.hh \
    MPEG2TransportStreamDemuxedTrack.hh \
    MPEG2TransportStreamParser.hh \
    MPEGVideoStreamParser.hh \
    MatroskaDemuxedTrack.hh \
    MatroskaFileParser.hh \
    MatroskaFileServerMediaSubsession.hh \
    OggDemuxedTrack.hh \
    OggFileParser.hh \
    OggFileServerMediaSubsession.hh \
    StreamParser.hh \
    include/AC3AudioFileServerMediaSubsession.hh \
    include/AC3AudioRTPSink.hh \
    include/AC3AudioRTPSource.hh \
    include/AC3AudioStreamFramer.hh \
    include/ADTSAudioFileServerMediaSubsession.hh \
    include/ADTSAudioFileSource.hh \
    include/AMRAudioFileServerMediaSubsession.hh \
    include/AMRAudioFileSink.hh \
    include/AMRAudioFileSource.hh \
    include/AMRAudioRTPSink.hh \
    include/AMRAudioRTPSource.hh \
    include/AMRAudioSource.hh \
    include/AVIFileSink.hh \
    include/AudioInputDevice.hh \
    include/AudioRTPSink.hh \
    include/Base64.hh \
    include/BasicUDPSink.hh \
    include/BasicUDPSource.hh \
    include/BitVector.hh \
    include/ByteStreamFileSource.hh \
    include/ByteStreamMemoryBufferSource.hh \
    include/ByteStreamMultiFileSource.hh \
    include/DVVideoFileServerMediaSubsession.hh \
    include/DVVideoRTPSink.hh \
    include/DVVideoRTPSource.hh \
    include/DVVideoStreamFramer.hh \
    include/DeviceSource.hh \
    include/DigestAuthentication.hh \
    include/FileServerMediaSubsession.hh \
    include/FileSink.hh \
    include/FramedFileSource.hh \
    include/FramedFilter.hh \
    include/FramedSource.hh \
    include/GSMAudioRTPSink.hh \
    include/GenericMediaServer.hh \
    include/H261VideoRTPSource.hh \
    include/H263plusVideoFileServerMediaSubsession.hh \
    include/H263plusVideoRTPSink.hh \
    include/H263plusVideoRTPSource.hh \
    include/H263plusVideoStreamFramer.hh \
    include/H264VideoFileServerMediaSubsession.hh \
    include/H264VideoFileSink.hh \
    include/H264VideoRTPSink.hh \
    include/H264VideoRTPSource.hh \
    include/H264VideoStreamDiscreteFramer.hh \
    include/H264VideoStreamFramer.hh \
    include/H264or5VideoFileSink.hh \
    include/H264or5VideoRTPSink.hh \
    include/H264or5VideoStreamDiscreteFramer.hh \
    include/H264or5VideoStreamFramer.hh \
    include/H265VideoFileServerMediaSubsession.hh \
    include/H265VideoFileSink.hh \
    include/H265VideoRTPSink.hh \
    include/H265VideoRTPSource.hh \
    include/H265VideoStreamDiscreteFramer.hh \
    include/H265VideoStreamFramer.hh \
    include/HLSSegmenter.hh \
    include/InputFile.hh \
    include/JPEG2000VideoRTPSink.hh \
    include/JPEG2000VideoRTPSource.hh \
    include/JPEGVideoRTPSink.hh \
    include/JPEGVideoRTPSource.hh \
    include/JPEGVideoSource.hh \
    include/Locale.hh \
    include/MP3ADU.hh \
    include/MP3ADURTPSink.hh \
    include/MP3ADURTPSource.hh \
    include/MP3ADUTranscoder.hh \
    include/MP3ADUinterleaving.hh \
    include/MP3AudioFileServerMediaSubsession.hh \
    include/MP3FileSource.hh \
    include/MP3Transcoder.hh \
    include/MPEG1or2AudioRTPSink.hh \
    include/MPEG1or2AudioRTPSource.hh \
    include/MPEG1or2AudioStreamFramer.hh \
    include/MPEG1or2Demux.hh \
    include/MPEG1or2DemuxedElementaryStream.hh \
    include/MPEG1or2DemuxedServerMediaSubsession.hh \
    include/MPEG1or2FileServerDemux.hh \
    include/MPEG1or2VideoFileServerMediaSubsession.hh \
    include/MPEG1or2VideoRTPSink.hh \
    include/MPEG1or2VideoRTPSource.hh \
    include/MPEG1or2VideoStreamDiscreteFramer.hh \
    include/MPEG1or2VideoStreamFramer.hh \
    include/MPEG2IndexFromTransportStream.hh \
    include/MPEG2TransportFileServerMediaSubsession.hh \
    include/MPEG2TransportStreamAccumulator.hh \
    include/MPEG2TransportStreamDemux.hh \
    include/MPEG2TransportStreamFramer.hh \
    include/MPEG2TransportStreamFromESSource.hh \
    include/MPEG2TransportStreamFromPESSource.hh \
    include/MPEG2TransportStreamIndexFile.hh \
    include/MPEG2TransportStreamMultiplexor.hh \
    include/MPEG2TransportStreamTrickModeFilter.hh \
    include/MPEG2TransportUDPServerMediaSubsession.hh \
    include/MPEG4ESVideoRTPSink.hh \
    include/MPEG4ESVideoRTPSource.hh \
    include/MPEG4GenericRTPSink.hh \
    include/MPEG4GenericRTPSource.hh \
    include/MPEG4LATMAudioRTPSink.hh \
    include/MPEG4LATMAudioRTPSource.hh \
    include/MPEG4VideoFileServerMediaSubsession.hh \
    include/MPEG4VideoStreamDiscreteFramer.hh \
    include/MPEG4VideoStreamFramer.hh \
    include/MPEGVideoStreamFramer.hh \
    include/MatroskaFile.hh \
    include/MatroskaFileServerDemux.hh \
    include/Media.hh \
    include/MediaSession.hh \
    include/MediaSink.hh \
    include/MediaSource.hh \
    include/MediaTranscodingTable.hh \
    include/MultiFramedRTPSink.hh \
    include/MultiFramedRTPSource.hh \
    include/OggFile.hh \
    include/OggFileServerDemux.hh \
    include/OggFileSink.hh \
    include/OnDemandServerMediaSubsession.hh \
    include/OutputFile.hh \
    include/PassiveServerMediaSubsession.hh \
    include/ProxyServerMediaSession.hh \
    include/QCELPAudioRTPSource.hh \
    include/QuickTimeFileSink.hh \
    include/QuickTimeGenericRTPSource.hh \
    include/RTCP.hh \
    include/RTPInterface.hh \
    include/RTPSink.hh \
    include/RTPSource.hh \
    include/RTSPClient.hh \
    include/RTSPCommon.hh \
    include/RTSPRegisterSender.hh \
    include/RTSPServer.hh \
    include/RawVideoRTPSink.hh \
    include/RawVideoRTPSource.hh \
    include/SIPClient.hh \
    include/ServerMediaSession.hh \
    include/SimpleRTPSink.hh \
    include/SimpleRTPSource.hh \
    include/StreamReplicator.hh \
    include/T140TextRTPSink.hh \
    include/TLSState.hh \
    include/TextRTPSink.hh \
    include/TheoraVideoRTPSink.hh \
    include/TheoraVideoRTPSource.hh \
    include/VP8VideoRTPSink.hh \
    include/VP8VideoRTPSource.hh \
    include/VP9VideoRTPSink.hh \
    include/VP9VideoRTPSource.hh \
    include/VideoRTPSink.hh \
    include/VorbisAudioRTPSink.hh \
    include/VorbisAudioRTPSource.hh \
    include/WAVAudioFileServerMediaSubsession.hh \
    include/WAVAudioFileSource.hh \
    include/liveMedia.hh \
    include/liveMedia_version.hh \
    include/ourMD5.hh \
    include/uLawAudioFilter.hh \
    rtcp_from_spec.h


# Default rules for deployment.
unix {
    target.path = $$[QT_INSTALL_PLUGINS]/generic
}
!isEmpty(target.path): INSTALLS += target
