QT += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = GeneticEngine
CONFIG += console
CONFIG += link_pkgconfig
CONFIG += c++11

TEMPLATE = app

SOURCES += main.cpp \
    genetictree.cpp \
    geneticengine.cpp \
    geneticprogram.cpp

PKGCONFIG += opencv

HEADERS += \
    genetictree.h \
    geneticengine.h \
    geneticprogram.h

