SET( SUBPROJECT_NAME ImageIO ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

MESSAGE(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
#FIND_PACKAGE(Boost 1.48.0 COMPONENTS filesystem REQUIRED)	# Patrik: I think we need 'system' too, if the app doesn't link to it, ImageIO fails to link.
FIND_PACKAGE(Boost 1.48.0 REQUIRED) # We only want the headers in the lib?
FIND_PACKAGE(OpenCV 2.3.1 REQUIRED core)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")
IF(WITH_MSKINECT_SDK)
	# Include Microsoft Kinect SDK (Windows)
	SET( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
	FIND_PACKAGE(KinectSDK REQUIRED)
	INCLUDE_DIRECTORIES(${KinectSDK_INCLUDE_DIR})
	add_definitions(-DWITH_MSKINECT_SDK)
ENDIF()

# source and header files
SET(HEADERS
	include/imageio/BobotLandmarkSink.hpp
	include/imageio/BobotLandmarkSource.hpp
	include/imageio/CameraImageSource.hpp
	include/imageio/DefaultNamedLandmarkSource.hpp
	include/imageio/DefaultLandmarkSink.hpp
	include/imageio/DidLandmarkFormatParser.hpp
	include/imageio/DirectoryImageSink.hpp
	include/imageio/DirectoryImageSource.hpp
	include/imageio/EmptyLandmarkSource.hpp
	include/imageio/FddbLandmarkSink.hpp
	include/imageio/FileImageSource.hpp
	include/imageio/FileListImageSource.hpp
	include/imageio/IbugLandmarkFormatParser.hpp
	include/imageio/ImageSink.hpp
	include/imageio/ImageSource.hpp
	include/imageio/KinectImageSource.hpp
	include/imageio/LabeledImageSource.hpp
	include/imageio/Landmark.hpp
	include/imageio/LandmarkCollection.hpp
	include/imageio/LandmarkFileGatherer.hpp
	include/imageio/LandmarkFormatParser.hpp
	include/imageio/LandmarksHelper.hpp
	include/imageio/LfpwLandmarkFormatParser.hpp
	include/imageio/LstLandmarkFormatParser.hpp
	include/imageio/ModelLandmark.hpp
	include/imageio/NamedLabeledImageSource.hpp
	include/imageio/NamedLandmarkSource.hpp
	include/imageio/OrderedLabeledImageSource.hpp
	include/imageio/OrderedLandmarkSink.hpp
	include/imageio/OrderedLandmarkSource.hpp
	include/imageio/RectLandmark.hpp
	include/imageio/RepeatingFileImageSource.hpp
	include/imageio/TlmsLandmarkFormatParser.hpp
	include/imageio/VideoImageSink.hpp
	include/imageio/VideoImageSource.hpp
)
SET(SOURCE
	src/imageio/BobotLandmarkSink.cpp
	src/imageio/BobotLandmarkSource.cpp
	src/imageio/CameraImageSource.cpp
	src/imageio/DefaultNamedLandmarkSource.cpp
	src/imageio/DefaultLandmarkSink.cpp
	src/imageio/DidLandmarkFormatParser.cpp
	src/imageio/DirectoryImageSink.cpp
	src/imageio/DirectoryImageSource.cpp
	src/imageio/FddbLandmarkSink.cpp
	src/imageio/FileImageSource.cpp
	src/imageio/FileListImageSource.cpp
	src/imageio/IbugLandmarkFormatParser.cpp
	src/imageio/KinectImageSource.cpp
	src/imageio/LandmarkCollection.cpp
	src/imageio/LandmarkFileGatherer.cpp
	src/imageio/LfpwLandmarkFormatParser.cpp
	src/imageio/LstLandmarkFormatParser.cpp
	src/imageio/ModelLandmark.cpp
	src/imageio/NamedLabeledImageSource.cpp
	src/imageio/OrderedLabeledImageSource.cpp
	src/imageio/RectLandmark.cpp
	src/imageio/RepeatingFileImageSource.cpp
	src/imageio/TlmsLandmarkFormatParser.cpp
	src/imageio/VideoImageSink.cpp
	src/imageio/VideoImageSource.cpp
)

include_directories( "include" )
include_directories( ${Logging_SOURCE_DIR}/include )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Boost_INCLUDE_DIRS} )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
