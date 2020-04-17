TEMPLATE = subdirs


SUBDIRS = BasicUsageEnvironment groupsock liveMedia UsageEnvironment \
    QtRTSPServer.pro

BasicUsageEnvironment.subdir = ../libs/LIVE555/BasicUsageEnvironment
groupsock.subdir = ../libs/LIVE555/groupsock
liveMedia.subdir = ../libs/LIVE555/liveMedia
UsageEnvironment.subdir = ../libs/LIVE555/UsageEnvironment
QtRTSPServer.subdir = ../Server

QtRTSPServer.pro.depends = BasicUsageEnvironment groupsock liveMedia UsageEnvironment

CONFIG += ordered

