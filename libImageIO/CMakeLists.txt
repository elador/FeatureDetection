SET( SUBPROJECT_NAME ImageIO ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
#FIND_PACKAGE(Boost 1.48.0 COMPONENTS filesystem REQUIRED)	# Patrik: I think we need 'system' too, if the app doesn't link to it, ImageIO fails to link.
FIND_PACKAGE(Boost 1.48.0 REQUIRED) # We only want the headers in the lib?
FIND_PACKAGE(OpenCV 2.3.1 REQUIRED core)
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
	include/imageio/ImageSource.hpp
	include/imageio/FileImageSource.hpp
	include/imageio/RepeatingFileImageSource.hpp
	include/imageio/FileListImageSource.hpp
	include/imageio/DirectoryImageSource.hpp
	include/imageio/VideoImageSource.hpp
	include/imageio/KinectImageSource.hpp
	include/imageio/NamedLabeledImageSource.hpp
	include/imageio/ImageSink.hpp
	include/imageio/DirectoryImageSink.hpp
	include/imageio/VideoImageSink.hpp
	include/imageio/Landmark.hpp
	include/imageio/LandmarkCollection.hpp
	include/imageio/OrderedLandmarkSource.hpp
	include/imageio/NamedLandmarkSource.hpp
	include/imageio/DefaultNamedLandmarkSource.hpp
	include/imageio/LandmarkFileGatherer.hpp
	include/imageio/LandmarkFormatParser.hpp
	include/imageio/DidLandmarkFormatParser.hpp
	include/imageio/TlmsLandmarkFormatParser.hpp
	include/imageio/LandmarksHelper.hpp
)
SET(SOURCE
	src/imageio/FileImageSource.cpp
	src/imageio/RepeatingFileImageSource.cpp
	src/imageio/FileListImageSource.cpp
	src/imageio/DirectoryImageSource.cpp
	src/imageio/VideoImageSource.cpp
	src/imageio/KinectImageSource.cpp
	src/imageio/NamedLabeledImageSource.cpp
	src/imageio/DirectoryImageSink.cpp
	src/imageio/VideoImageSink.cpp
	src/imageio/Landmark.cpp
	src/imageio/LandmarkCollection.cpp
	src/imageio/DefaultNamedLandmarkSource.cpp
	src/imageio/LandmarkFileGatherer.cpp
	src/imageio/DidLandmarkFormatParser.cpp
	src/imageio/TlmsLandmarkFormatParser.cpp
)

include_directories( "include" )
include_directories( ${Logging_SOURCE_DIR}/include )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Boost_INCLUDE_DIRS} )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
