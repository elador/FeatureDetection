SET( SUBPROJECT_NAME ImageProcessing ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(OpenCV REQUIRED core imgproc)

# source and header files
SET(HEADERS
	include/imageprocessing/FeatureExtractor.hpp
	include/imageprocessing/FeatureTransformer.hpp
	include/imageprocessing/FilteringFeatureTransformer.hpp
	include/imageprocessing/GrayscaleFilter.hpp
	include/imageprocessing/HistogramEqualizationFilter.hpp
	include/imageprocessing/IdentityFeatureTransformer.hpp
	include/imageprocessing/ImageFilter.hpp
	include/imageprocessing/ImagePyramid.hpp
	include/imageprocessing/ImagePyramidLayer.hpp
	include/imageprocessing/IntegralImageFilter.hpp
	include/imageprocessing/MultipleImageFilter.hpp
	include/imageprocessing/Patch.hpp
	include/imageprocessing/PatchExtractor.hpp
	include/imageprocessing/PyramidPatchExtractor.hpp
	include/imageprocessing/RowFeatureTransformer.hpp
	include/imageprocessing/WhiteningFilter.hpp
	include/imageprocessing/ZeroMeanUnitVarianceFilter.hpp
	
)
SET(SOURCE
	src/imageprocessing/GrayscaleFilter.cpp
	src/imageprocessing/HistogramEqualizationFilter.cpp
	src/imageprocessing/ImagePyramid.cpp
	src/imageprocessing/ImagePyramidLayer.cpp
	src/imageprocessing/IntegralImageFilter.cpp
	src/imageprocessing/MultipleImageFilter.cpp
	src/imageprocessing/PyramidPatchExtractor.cpp
	src/imageprocessing/WhiteningFilter.cpp
	src/imageprocessing/ZeroMeanUnitVarianceFilter.cpp
)

include_directories( "include" )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
