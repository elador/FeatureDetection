SET( SUBPROJECT_NAME faceTrackingApp )	# my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(Boost COMPONENTS random system filesystem program_options REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED core imgproc highgui)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")
# Matlab
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
FIND_PACKAGE(MATLAB REQUIRED)

IF(WIN32)
	# In Windows only: Include Microsoft Kinect SDK
	SET( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
	FIND_PACKAGE(KinectSDK REQUIRED)
	INCLUDE_DIRECTORIES(${KinectSDK_INCLUDE_DIR})
	SET(KINECT_LIBNAME ${KinectSDK_LIBRARIES})
	
	# Fix for VS2012 that has _VARIADIC_MAX set to 5 (the C++11 standard requires a value
	# of 10). Don't set too high because it increases compiler memory usage / compile-time.
	add_definitions( -D_VARIADIC_MAX=6 )
ELSE()
	SET(KINECT_LIBNAME "")
ENDIF()

# source and header files
SET(HEADERS
	FaceTracking.h
	wingettimeofday.h
)
SET(SOURCE
	FaceTracking.cpp
	wingettimeofday.cpp
)

# add dependencies
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Logging_SOURCE_DIR}/include )
include_directories( ${ImageIO_SOURCE_DIR}/include )
include_directories( ${ImageProcessing_SOURCE_DIR}/include )
include_directories( ${Classification_SOURCE_DIR}/include )
include_directories( ${Detection_SOURCE_DIR}/include )
include_directories( ${Condensation_SOURCE_DIR}/include )

# make executable
ADD_EXECUTABLE( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME} ${Boost_LIBRARIES} ${MATLAB_MX_LIBRARY} ${MATLAB_MAT_LIBRARY} ${OpenCV_LIBS} Logging ImageIO ${KINECT_LIBNAME} ImageProcessing Classification Detection Condensation )
