SET( SUBPROJECT_NAME VideoPlayerApp ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(Boost 1.48.0 COMPONENTS system filesystem REQUIRED)
FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core highgui)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

IF(WITH_MSKINECT_SDK)
	# Include Microsoft Kinect SDK (Windows)
	SET( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
	FIND_PACKAGE(KinectSDK REQUIRED)
	INCLUDE_DIRECTORIES(${KinectSDK_INCLUDE_DIR})
	SET(KINECT_LIBNAME ${KinectSDK_LIBRARIES})
	add_definitions(-DWITH_MSKINECT_SDK)
ELSE()
	SET(KINECT_LIBNAME "")
ENDIF()

IF(WIN32)
	# Fix for VS2012 that has _VARIADIC_MAX set to 5 (the C++11 standard requires a value
	# of 10). Don't set too high because it increases compiler memory usage / compile-time.
	add_definitions( -D_VARIADIC_MAX=9 ) # Todo Might not be needed anymore in VS2013
ENDIF()

# source and header files
SET(HEADERS
	VideoPlayer.hpp
)
SET(SOURCE
	VideoPlayer.cpp
)

# add dependencies
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Logging_SOURCE_DIR}/include )
include_directories( ${ImageIO_SOURCE_DIR}/include )

# make executable
ADD_EXECUTABLE( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME} ImageIO ${KINECT_LIBNAME} Logging ${OpenCV_LIBS} ${Boost_LIBRARIES} )
