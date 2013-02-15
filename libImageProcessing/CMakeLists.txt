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
	include/ImagePyramid.hpp
	include/PyramidLayer.hpp
	include/Patch.hpp
	include/PatchExtractor.hpp
	include/FeatureTransformer.hpp
	include/RowFeatureTransformer.hpp
	include/ImageFilter.hpp
	
)
SET(SOURCE
	src/ImagePyramid.cpp
	src/PyramidLayer.cpp
	src/Patch.cpp
	src/PatchExtractor.cpp
	src/RowFeatureTransformer.cpp
)

include_directories( "include" )

# add dependencies
#include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${libFeatureDetection_SOURCE_DIR} )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
