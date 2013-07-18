SET( SUBPROJECT_NAME ImageProcessing ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

MESSAGE(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

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
	include/imageprocessing/DirectImageFeatureExtractor.hpp
	include/imageprocessing/DirectPyramidFeatureExtractor.hpp
	include/imageprocessing/FeatureExtractor.hpp
	include/imageprocessing/FilteringFeatureExtractor.hpp
	include/imageprocessing/FilteringPyramidFeatureExtractor.hpp
	include/imageprocessing/GradientFilter.hpp
	include/imageprocessing/GradientHistogramFilter.hpp
	include/imageprocessing/GradientMagnitudeFilter.hpp
	include/imageprocessing/GrayscaleFilter.hpp
	include/imageprocessing/HaarFeatureFilter.hpp
	include/imageprocessing/HistEq64Filter.hpp
	include/imageprocessing/HistogramEqualizationFilter.hpp
	include/imageprocessing/HistogramFeatureExtractor.hpp
	include/imageprocessing/HistogramFilter.hpp
	include/imageprocessing/ImageFilter.hpp
	include/imageprocessing/ImagePyramid.hpp
	include/imageprocessing/ImagePyramidLayer.hpp
	include/imageprocessing/IntegralImageFilter.hpp
	include/imageprocessing/IntensityNormNormalizationFilter.hpp
	include/imageprocessing/LbpFilter.hpp
	include/imageprocessing/MultipleImageFilter.hpp
	include/imageprocessing/Patch.hpp
	include/imageprocessing/PatchResizingFeatureExtractor.hpp
	include/imageprocessing/PyramidFeatureExtractor.hpp
	include/imageprocessing/ReshapingFilter.hpp
	include/imageprocessing/ResizingFilter.hpp
	include/imageprocessing/SpatialHistogramFeatureExtractor.hpp
	include/imageprocessing/SpatialPyramidHistogramFeatureExtractor.hpp
	include/imageprocessing/VersionedImage.hpp
	include/imageprocessing/WhiteningFilter.hpp
	include/imageprocessing/ZeroMeanUnitVarianceFilter.hpp
)
SET(SOURCE
	src/imageprocessing/CachingFeatureExtractor.cpp
	src/imageprocessing/CachingPyramidFeatureExtractor.cpp
	src/imageprocessing/ConversionFilter.cpp
	src/imageprocessing/DirectImageFeatureExtractor.cpp
	src/imageprocessing/DirectPyramidFeatureExtractor.cpp
	src/imageprocessing/GradientFilter.cpp
	src/imageprocessing/GradientHistogramFilter.cpp
	src/imageprocessing/GradientMagnitudeFilter.cpp
	src/imageprocessing/GrayscaleFilter.cpp
	src/imageprocessing/HaarFeatureFilter.cpp
	src/imageprocessing/HistEq64Filter.cpp
	src/imageprocessing/HistogramEqualizationFilter.cpp
	src/imageprocessing/HistogramFeatureExtractor.cpp
	src/imageprocessing/ImagePyramid.cpp
	src/imageprocessing/ImagePyramidLayer.cpp
	src/imageprocessing/IntegralImageFilter.cpp
	src/imageprocessing/IntensityNormNormalizationFilter.cpp
	src/imageprocessing/LbpFilter.cpp
	src/imageprocessing/MultipleImageFilter.cpp
	src/imageprocessing/ReshapingFilter.cpp
	src/imageprocessing/ResizingFilter.cpp
	src/imageprocessing/SpatialHistogramFeatureExtractor.cpp
	src/imageprocessing/SpatialPyramidHistogramFeatureExtractor.cpp
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
