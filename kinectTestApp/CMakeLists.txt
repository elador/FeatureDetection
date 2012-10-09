SET( SUBPROJECT_NAME kinectTestApp )	# my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(Boost REQUIRED random system filesystem)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
FIND_PACKAGE(OpenCV REQUIRED core imgproc highgui)
# Matlab
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
FIND_PACKAGE(MATLAB REQUIRED)

# In Windows only: Include Microsoft Kinect SDK
FIND_PACKAGE(KinectSDK REQUIRED)
INCLUDE_DIRECTORIES(${KinectSDK_INCLUDE_DIR})

# source and header files
SET(HEADERS
	DirectoryImageSource.h
	FaceTracking.h
	ImageSource.h
	KinectImageSource.h
	wingettimeofday.h
)
SET(SOURCE
	DirectoryImageSource.cpp
	FaceTracking.cpp
	KinectImageSource.cpp
	wingettimeofday.cpp
)

# add dependencies
include_directories( ${Tracking_SOURCE_DIR}/include )
include_directories( ${libFeatureDetection_SOURCE_DIR} )

# make executable
ADD_EXECUTABLE( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
TARGET_LINK_LIBRARIES( ${SUBPROJECT_NAME} Tracking libFeatureDetection ${OpenCV_LIBS} ${Boost_LIBRARIES} ${MATLAB_MX_LIBRARY} ${MATLAB_MAT_LIBRARY} ${KinectSDK_LIBRARIES} )
