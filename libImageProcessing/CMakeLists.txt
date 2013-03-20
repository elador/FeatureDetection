SET( SUBPROJECT_NAME ImageProcessing ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(OpenCV REQUIRED core imgproc)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

# source and header files
SET(HEADERS
	include/imageprocessing/ConversionFilter.hpp
	include/imageprocessing/FeatureExtractor.hpp
	include/imageprocessing/FilteringFeatureExtractor.hpp
	include/imageprocessing/GrayscaleFilter.hpp
	include/imageprocessing/HistogramEqualizationFilter.hpp
	include/imageprocessing/HistEq64Filter.hpp
	include/imageprocessing/ImageFilter.hpp
	include/imageprocessing/ImagePyramid.hpp
	include/imageprocessing/ImagePyramidLayer.hpp
	include/imageprocessing/IntegralImageFilter.hpp
	include/imageprocessing/MultipleImageFilter.hpp
	include/imageprocessing/Patch.hpp
	include/imageprocessing/PatchExtractor.hpp
	include/imageprocessing/PyramidFeatureExtractor.hpp
	include/imageprocessing/ResizingFilter.hpp
	include/imageprocessing/VersionedImage.hpp
	include/imageprocessing/WhiteningFilter.hpp
	include/imageprocessing/ZeroMeanUnitVarianceFilter.hpp
	
)
SET(SOURCE
	src/imageprocessing/ConversionFilter.cpp
	src/imageprocessing/GrayscaleFilter.cpp
	src/imageprocessing/HistEq64Filter.cpp
	src/imageprocessing/HistogramEqualizationFilter.cpp
	src/imageprocessing/ImagePyramid.cpp
	src/imageprocessing/ImagePyramidLayer.cpp
	src/imageprocessing/IntegralImageFilter.cpp
	src/imageprocessing/MultipleImageFilter.cpp
	src/imageprocessing/PyramidFeatureExtractor.cpp
	src/imageprocessing/ResizingFilter.cpp
	src/imageprocessing/WhiteningFilter.cpp
	src/imageprocessing/ZeroMeanUnitVarianceFilter.cpp
)

include_directories( "include" )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
