set(SUBPROJECT_NAME ImageIO)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
find_package(Boost 1.48.0 COMPONENTS system filesystem REQUIRED)

find_package(OpenCV 2.4.3 REQUIRED core highgui)

if(WITH_MSKINECT_SDK)
	# Include Microsoft Kinect SDK (Windows)
	set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
	find_package(KinectSDK REQUIRED)
	include_directories(${KinectSDK_INCLUDE_DIR})
	set(KINECT_LIBNAME ${KinectSDK_LIBRARIES})
	add_definitions(-DWITH_MSKINECT_SDK)
else()
	set(KINECT_LIBNAME "")
endif()

# source and header files
set(HEADERS
	include/imageio/BobotLandmarkSink.hpp
	include/imageio/BobotLandmarkSource.hpp
	include/imageio/CameraImageSource.hpp
	include/imageio/DefaultNamedLandmarkSource.hpp
	include/imageio/DidLandmarkFormatParser.hpp
	include/imageio/DidLandmarkSink.hpp
	include/imageio/DirectoryImageSink.hpp
	include/imageio/DirectoryImageSource.hpp
	include/imageio/DlibImageSource.hpp
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
	include/imageio/LandmarkMapper.hpp
	include/imageio/LandmarkSource.hpp
	include/imageio/LandmarksHelper.hpp
	include/imageio/LfpwLandmarkFormatParser.hpp
	include/imageio/LstLandmarkFormatParser.hpp
	include/imageio/ModelLandmark.hpp
	include/imageio/MuctLandmarkFormatParser.hpp
	include/imageio/NamedLabeledImageSource.hpp
	include/imageio/NamedLandmarkSink.hpp
	include/imageio/NamedLandmarkSource.hpp
	include/imageio/OrderedLabeledImageSource.hpp
	include/imageio/OrderedLandmarkSink.hpp
	include/imageio/PascStillEyesLandmarkFormatParser.hpp
	include/imageio/RectLandmark.hpp
	include/imageio/RectLandmarkSink.hpp
	include/imageio/RepeatingFileImageSource.hpp
	include/imageio/SingleLandmarkSink.hpp
	include/imageio/SingleLandmarkSource.hpp
	include/imageio/SimpleModelLandmarkFormatParser.hpp
	include/imageio/SimpleModelLandmarkSink.hpp
	include/imageio/SimpleRectLandmarkFormatParser.hpp
	include/imageio/TlmsLandmarkFormatParser.hpp
	include/imageio/VideoImageSink.hpp
	include/imageio/VideoImageSource.hpp
)
set(SOURCE
	src/imageio/BobotLandmarkSink.cpp
	src/imageio/BobotLandmarkSource.cpp
	src/imageio/CameraImageSource.cpp
	src/imageio/DefaultNamedLandmarkSource.cpp
	src/imageio/DidLandmarkFormatParser.cpp
	src/imageio/DidLandmarkSink.cpp
	src/imageio/DirectoryImageSink.cpp
	src/imageio/DirectoryImageSource.cpp
	src/imageio/DlibImageSource.cpp
	src/imageio/FddbLandmarkSink.cpp
	src/imageio/FileImageSource.cpp
	src/imageio/FileListImageSource.cpp
	src/imageio/IbugLandmarkFormatParser.cpp
	src/imageio/KinectImageSource.cpp
	src/imageio/LandmarkCollection.cpp
	src/imageio/LandmarkFileGatherer.cpp
	src/imageio/LandmarkMapper.cpp
	src/imageio/LfpwLandmarkFormatParser.cpp
	src/imageio/LstLandmarkFormatParser.cpp
	src/imageio/ModelLandmark.cpp
	src/imageio/MuctLandmarkFormatParser.cpp
	src/imageio/NamedLabeledImageSource.cpp
	src/imageio/OrderedLabeledImageSource.cpp
	src/imageio/PascStillEyesLandmarkFormatParser.cpp
	src/imageio/RectLandmark.cpp
	src/imageio/RectLandmarkSink.cpp
	src/imageio/RepeatingFileImageSource.cpp
	src/imageio/SingleLandmarkSink.cpp
	src/imageio/SingleLandmarkSource.cpp
	src/imageio/SimpleModelLandmarkFormatParser.cpp
	src/imageio/SimpleModelLandmarkSink.cpp
	src/imageio/SimpleRectLandmarkFormatParser.cpp
	src/imageio/TlmsLandmarkFormatParser.cpp
	src/imageio/VideoImageSink.cpp
	src/imageio/VideoImageSource.cpp
)

include_directories("include")
include_directories(${Logging_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# make library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} Logging ${KINECT_LIBNAME} ${Boost_LIBRARIES} ${OpenCV_LIBS})
