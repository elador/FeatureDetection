SET( SUBPROJECT_NAME ImageIO ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(Boost COMPONENTS filesystem REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED core)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")
IF(WIN32)
	# In Windows only: Include Microsoft Kinect SDK
	SET( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
	FIND_PACKAGE(KinectSDK REQUIRED)
	INCLUDE_DIRECTORIES(${KinectSDK_INCLUDE_DIR})
ENDIF()


# source and header files
SET(HEADERS
	include/imageio/DirectoryImageSink.hpp
	include/imageio/DirectoryImageSource.hpp
	include/imageio/ImageSink.hpp
	include/imageio/ImageSource.hpp
	include/imageio/KinectImageSource.hpp
	include/imageio/VideoImageSink.hpp
	include/imageio/VideoImageSource.hpp
)
SET(SOURCE
	src/imageio/DirectoryImageSink.cpp
	src/imageio/DirectoryImageSource.cpp
	src/imageio/KinectImageSource.cpp
	src/imageio/VideoImageSink.cpp
	src/imageio/VideoImageSource.cpp
)

include_directories( "include" )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Boost_INCLUDE_DIRS} )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
