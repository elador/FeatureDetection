SET( SUBPROJECT_NAME SVM ) # my own variable, not cmake's
PROJECT( ${SUBPROJECT_NAME} )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET( ${SUBPROJECT_NAME}_VERSION_MAJOR 0 )
SET( ${SUBPROJECT_NAME}_VERSION_MINOR 1 )

message(STATUS "=== Configuring ${SUBPROJECT_NAME} ===")

# find dependencies
FIND_PACKAGE(OpenCV 2.4.3 REQUIRED core)

# source and header files
SET(HEADERS
	include/svm.h
	include/libsvm/LibSvmClassifier.hpp
	include/libsvm/LibSvmKernelParamSetter.hpp
	include/libsvm/LibSvmUtils.hpp
)
SET(SOURCE
	src/svm.cpp
	src/libsvm/LibSvmClassifier.cpp
	src/libsvm/LibSvmUtils.cpp
)

include_directories( "include" )

# add dependencies
include_directories(${Classification_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# make library
add_library(${SUBPROJECT_NAME} ${SOURCE} ${HEADERS})
target_link_libraries(${SUBPROJECT_NAME} Classification ${OpenCV_LIBS})
