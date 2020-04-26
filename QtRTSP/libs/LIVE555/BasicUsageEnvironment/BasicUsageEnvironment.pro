QT -= gui core

TEMPLATE = lib
CONFIG += staticlib

CONFIG += c++11

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

SOURCES += BasicHashTable.cpp \
    BasicTaskScheduler.cpp \
    BasicTaskScheduler0.cpp \
    BasicUsageEnvironment.cpp \
    BasicUsageEnvironment0.cpp \
    DelayQueue.cpp


HEADERS += \
    include/BasicHashTable.hh \
    include/BasicHashTable.hh \
    include/BasicUsageEnvironment.hh \
    include/BasicUsageEnvironment0.hh \
    include/BasicUsageEnvironment_version.hh \
    include/DelayQueue.hh \
    include/HandlerSet.hh

# Default rules for deployment.
unix {
    target.path = $$[QT_INSTALL_PLUGINS]/generic
}
!isEmpty(target.path): INSTALLS += target
