SET( SUBPROJECT_NAME ImageProcessing ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(Boost 1.48.0 REQUIRED) # We only need the headers

FIND_PACKAGE(OpenCV 2.3.1 REQUIRED core imgproc)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

# source and header files
SET(HEADERS
	include/imageprocessing/CachingFeatureExtractor.hpp
	include/imageprocessing/CachingPyramidFeatureExtractor.hpp
	include/imageprocessing/ConversionFilter.hpp
	include/imageprocessing/DirectPyramidFeatureExtractor.hpp
	include/imageprocessing/FeatureExtractor.hpp
	include/imageprocessing/FilteringFeatureExtractor.hpp
	include/imageprocessing/FilteringPyramidFeatureExtractor.hpp
	include/imageprocessing/GrayscaleFilter.hpp
	include/imageprocessing/HistogramEqualizationFilter.hpp
	include/imageprocessing/HistEq64Filter.hpp
	include/imageprocessing/ImageFilter.hpp
	include/imageprocessing/ImagePyramid.hpp
	include/imageprocessing/ImagePyramidLayer.hpp
	include/imageprocessing/IntegralImageFilter.hpp
	include/imageprocessing/MultipleImageFilter.hpp
	include/imageprocessing/Patch.hpp
	include/imageprocessing/PyramidFeatureExtractor.hpp
	include/imageprocessing/ReshapingFilter.hpp
	include/imageprocessing/ResizingFilter.hpp
	include/imageprocessing/VersionedImage.hpp
	include/imageprocessing/WhiteningFilter.hpp
	include/imageprocessing/ZeroMeanUnitVarianceFilter.hpp
	
)
SET(SOURCE
	src/imageprocessing/CachingFeatureExtractor.cpp
	src/imageprocessing/CachingPyramidFeatureExtractor.cpp
	src/imageprocessing/ConversionFilter.cpp
	src/imageprocessing/DirectPyramidFeatureExtractor.cpp
	src/imageprocessing/GrayscaleFilter.cpp
	src/imageprocessing/HistEq64Filter.cpp
	src/imageprocessing/HistogramEqualizationFilter.cpp
	src/imageprocessing/ImagePyramid.cpp
	src/imageprocessing/ImagePyramidLayer.cpp
	src/imageprocessing/IntegralImageFilter.cpp
	src/imageprocessing/MultipleImageFilter.cpp
	src/imageprocessing/ReshapingFilter.cpp
	src/imageprocessing/ResizingFilter.cpp
	src/imageprocessing/WhiteningFilter.cpp
	src/imageprocessing/ZeroMeanUnitVarianceFilter.cpp
)

include_directories( "include" )

# add dependencies
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${Logging_SOURCE_DIR}/include )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
