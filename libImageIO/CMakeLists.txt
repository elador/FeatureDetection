set(SUBPROJECT_NAME ImageIO)
project(${SUBPROJECT_NAME})
cmake_minimum_required(VERSION 2.8)
set(${SUBPROJECT_NAME}_VERSION_MAJOR 0)
set(${SUBPROJECT_NAME}_VERSION_MINOR 1)

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
find_package(Boost 1.48.0 COMPONENTS system filesystem REQUIRED)

find_package(OpenCV 2.4.3 REQUIRED core highgui)

# source and header files
set(HEADERS
	include/imageio/AnnotatedImageSource.hpp
	include/imageio/AnnotationSink.hpp
	include/imageio/AnnotationSource.hpp
	include/imageio/BobotAnnotationSink.hpp
	include/imageio/BobotAnnotationSource.hpp
	include/imageio/CameraImageSource.hpp
	include/imageio/DirectoryImageSink.hpp
	include/imageio/DirectoryImageSource.hpp
	include/imageio/EmptyAnnotationSource.hpp
	include/imageio/ImageSink.hpp
	include/imageio/ImageSource.hpp
	include/imageio/SimpleAnnotationSink.hpp
	include/imageio/SimpleAnnotationSource.hpp
	include/imageio/VideoImageSink.hpp
	include/imageio/VideoImageSource.hpp
)
set(SOURCE
	src/imageio/AnnotatedImageSource.cpp
	src/imageio/BobotAnnotationSink.cpp
	src/imageio/BobotAnnotationSource.cpp
	src/imageio/CameraImageSource.cpp
	src/imageio/DirectoryImageSink.cpp
	src/imageio/DirectoryImageSource.cpp
	src/imageio/SimpleAnnotationSink.cpp
	src/imageio/SimpleAnnotationSource.cpp
	src/imageio/VideoImageSink.cpp
	src/imageio/VideoImageSource.cpp
)

include_directories("include")
include_directories(${Logging_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# make library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} Logging ${Boost_LIBRARIES} ${OpenCV_LIBS})
