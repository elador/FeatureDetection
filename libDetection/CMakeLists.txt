SET( SUBPROJECT_NAME Detection ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

# find dependencies
FIND_PACKAGE(Boost REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED core)

# source and header files
SET(HEADERS
	include/detection/Detector.hpp
	include/detection/SlidingWindowDetector.hpp
	include/detection/CascadeWvmOeSvmOe.h
	include/detection/CascadeWvmSvm.h
	include/detection/CircleDetector.h
	include/detection/DetectorSVM.h
	include/detection/DetectorWVM.h
	include/detection/FdPatch.h
	include/detection/FdPoint.h
	include/detection/OverlapElimination.h
	include/detection/Rect.h
	include/detection/RegressorSVR.h
	include/detection/RegressorWVR.h
	include/detection/SkinDetector.h
	include/detection/VDetector.h
	include/detection/VDetectorVectorMachine.h
)
SET(SOURCE
	src/detection/SlidingWindowDetector.cpp
	src/detection/CascadeWvmOeSvmOe.cpp
	src/detection/CascadeWvmSvm.cpp
	src/detection/CircleDetector.cpp
	src/detection/DetectorSVM.cpp
	src/detection/DetectorWVM.cpp
	src/detection/FdPatch.cpp
	src/detection/FdPoint.cpp
	src/detection/OverlapElimination.cpp
	src/detection/Rect.cpp
	src/detection/RegressorSVR.cpp
	src/detection/RegressorWVR.cpp
	src/detection/SkinDetector.cpp
	src/detection/VDetector.cpp
	src/detection/VDetectorVectorMachine.cpp
)

include_directories( "include" )

# add dependencies
include_directories( ${Boost_INCLUDE_DIRS} )
#include_directories( ${libFeatureDetection_SOURCE_DIR} )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
