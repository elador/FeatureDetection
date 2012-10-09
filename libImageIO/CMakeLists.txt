SET( SUBPROJECT_NAME ImageIO ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(Boost COMPONENTS filesystem REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED core)
IF(WIN32)
	# In Windows only: Include Microsoft Kinect SDK
	SET( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
	FIND_PACKAGE(KinectSDK REQUIRED)
	INCLUDE_DIRECTORIES(${KinectSDK_INCLUDE_DIR})
ENDIF()


# source and header files
SET(HEADERS
	include/imageio/ImageSource.h
	include/imageio/DirectoryImageSource.h
	include/imageio/VideoImageSource.h
	include/imageio/KinectImageSource.h
)
SET(SOURCE
	src/imageio/DirectoryImageSource.cpp
	src/imageio/VideoImageSource.cpp
	src/imageio/KinectImageSource.cpp
)

include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( "include" )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
