SET( SUBPROJECT_NAME Linear ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

MESSAGE(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(OpenCV 2.3.1 REQUIRED core)
MESSAGE(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

# source and header files
SET(HEADERS
	include/tron.h
	include/linear.h
	include/liblinear/LibLinearClassifier.hpp
	include/liblinear/LibLinearUtils.hpp
)
SET(SOURCE
	src/tron.cpp
	src/linear.cpp
	src/liblinear/LibLinearClassifier.cpp
	src/liblinear/LibLinearUtils.cpp
)

include_directories( "include" )

# add dependencies
include_directories( ${Classification_SOURCE_DIR}/include )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${BLAS_INCLUDE_DIRS} )

# make library
add_library( ${SUBPROJECT_NAME} ${SOURCE} ${HEADERS} )
