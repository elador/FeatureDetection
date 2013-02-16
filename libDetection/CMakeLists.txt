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
	include/detection/CascadeWvmOeSvmOe.cpp
	include/detection/CascadeWvmSvm.cpp
	include/detection/CircleDetector.cpp
	include/detection/DetectorSVM.cpp
	include/detection/DetectorWVM.cpp
	include/detection/FdPatch.cpp
	include/detection/FdPoint.cpp
	include/detection/OverlapElimination.cpp
	include/detection/Rect.cpp
	include/detection/RegressorSVR.cpp
	include/detection/RegressorWVR.cpp
	include/detection/SkinDetector.cpp
	include/detection/VDetector.cpp
	include/detection/VDetectorVectorMachine.cpp
)
SET(SOURCE
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
