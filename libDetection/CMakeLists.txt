SET( SUBPROJECT_NAME Detection ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
#FIND_PACKAGE(Boost REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED core)

# source and header files
SET(HEADERS
	include/detection/ClassifiedPatch.hpp
	include/detection/Detector.hpp
	include/detection/SlidingWindowDetector.hpp
	#include/detection/OverlapElimination.hpp
)
SET(SOURCE
	src/detection/SlidingWindowDetector.cpp
	#src/detection/OverlapElimination.cpp
)

include_directories( "include" )

# add dependencies
include_directories( ${Classification_SOURCE_DIR}/include )
include_directories( ${ImageProcessing_SOURCE_DIR}/include )
#include_directories( ${Boost_INCLUDE_DIRS} )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
