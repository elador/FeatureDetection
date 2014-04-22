set(SUBPROJECT_NAME softwareRendererTestApp)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8.11)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies:
FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core imgproc highgui calib3d)

FIND_PACKAGE(Boost 1.48.0 COMPONENTS program_options system filesystem REQUIRED)
if(Boost_FOUND)
  MESSAGE(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
ELSE(Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost not found")
ENDIF()

set(CMAKE_AUTOMOC ON)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
message(STATUS "Qt5 EGL include dir: ${Qt5Gui_EGL_INCLUDE_DIRS}")
message(STATUS "Qt5 OpenGL include dir: ${Qt5Gui_OPENGL_INCLUDE_DIRS}")
message(STATUS "Qt5 EGL library: ${Qt5Gui_EGL_LIBRARIES}")
message(STATUS "Qt5 OpenGL library: ${Qt5Gui_OPENGL_LIBRARIES}")

#Source and header files:
SET(SOURCE
	softwareRendererTestApp.cpp
	OpenGLWindow.cpp
	FittingWindow.cpp
)

SET(HEADERS
	OpenGLWindow.hpp
	FittingWindow.hpp
)

ADD_EXECUTABLE( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )

include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${HDF5_INCLUDE_DIRS} )
include_directories( ${Logging_SOURCE_DIR}/include )
include_directories( ${ImageIO_SOURCE_DIR}/include )
include_directories( ${Render_SOURCE_DIR}/include )
include_directories( ${MorphableModel_SOURCE_DIR}/include )

#Make the app depend on the libraries
TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME} ImageIO ${KINECT_LIBNAME} MorphableModel Render Logging ${Boost_LIBRARIES} ${OpenCV_LIBS} ${HDF5_LIBRARIES} Qt5::Core Qt5::Gui ${Qt5Gui_EGL_LIBRARIES} ${Qt5Gui_OPENGL_LIBRARIES} )
