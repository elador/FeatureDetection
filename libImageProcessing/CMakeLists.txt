SET( SUBPROJECT_NAME ImageProcessing ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core imgproc)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

# source and header files
SET(HEADERS
	include/imageprocessing/CellBasedPyramidFeatureExtractor.hpp
	include/imageprocessing/ChainedFilter.hpp
	include/imageprocessing/ConvolutionFilter.hpp
	include/imageprocessing/ExtendedHogFeatureExtractor.hpp
	include/imageprocessing/ExtendedHogFilter.hpp
	include/imageprocessing/FeatureExtractor.hpp
	include/imageprocessing/GradientFilter.hpp
	include/imageprocessing/GrayscaleFilter.hpp
	include/imageprocessing/ImageFilter.hpp
	include/imageprocessing/ImagePyramid.hpp
	include/imageprocessing/ImagePyramidLayer.hpp
	include/imageprocessing/Patch.hpp
	include/imageprocessing/PyramidFeatureExtractor.hpp
	include/imageprocessing/VersionedImage.hpp
)
SET(SOURCE
	src/imageprocessing/CellBasedPyramidFeatureExtractor.cpp
	src/imageprocessing/ChainedFilter.cpp
	src/imageprocessing/ConvolutionFilter.cpp
	src/imageprocessing/ExtendedHogFeatureExtractor.cpp
	src/imageprocessing/ExtendedHogFilter.cpp
	src/imageprocessing/GradientFilter.cpp
	src/imageprocessing/GrayscaleFilter.cpp
	src/imageprocessing/ImagePyramid.cpp
	src/imageprocessing/ImagePyramidLayer.cpp
	src/imageprocessing/PyramidFeatureExtractor.cpp
)

include_directories( "include" )

# add dependencies
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Logging_SOURCE_DIR}/include )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
target_link_libraries(${SUBPROJECT_NAME} Logging ${OpenCV_LIBS})
