SET( SUBPROJECT_NAME ImageProcessing ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
#FIND_PACKAGE(Boost REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED core)

# source and header files
SET(HEADERS
	include/imageprocessing/ImagePyramid.hpp
	include/imageprocessing/PyramidLayer.hpp
	include/imageprocessing/Patch.hpp
	include/imageprocessing/PatchExtractor.hpp
	include/imageprocessing/FeatureTransformer.hpp
	include/imageprocessing/RowFeatureTransformer.hpp
	include/imageprocessing/ImageFilter.hpp
	include/imageprocessing/FdImage.h
	include/imageprocessing/FeatureExtractor.h
	include/imageprocessing/HistEqFeatureTransformer.h
	include/imageprocessing/IImg.h
	include/imageprocessing/ImagePyramidFeatureExtractor_old.h
	include/imageprocessing/Pyramid.h
	include/imageprocessing/StdImage.h
	
)
SET(SOURCE
	src/imageprocessing/ImagePyramid.cpp
	src/imageprocessing/PyramidLayer.cpp
	src/imageprocessing/Patch.cpp
	src/imageprocessing/PatchExtractor.cpp
	src/imageprocessing/RowFeatureTransformer.cpp
	src/imageprocessing/ImagePyramidFeatureExtractor_old.cpp
	src/imageprocessing/IImg.cpp
	src/imageprocessing/HistEqFeatureTransformer.cpp
	src/imageprocessing/FdImage.cpp
	src/imageprocessing/Pyramid.cpp
	src/imageprocessing/StdImage.cpp
)

include_directories( "include" )

# add dependencies
#include_directories( ${Boost_INCLUDE_DIRS} )
#include_directories( ${libFeatureDetection_SOURCE_DIR} )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
